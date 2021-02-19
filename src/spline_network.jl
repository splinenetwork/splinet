using Flux, Zygote, Random
using Base: @kwdef
using BenchmarkTools
using LinearAlgebra
using UnsafeArrays

include("bsplines.jl")


# Define network types Spline(degree) and Resnet
struct NetworkType{T,D} end
Spline(D) = NetworkType{:spline,D}()
const Resnet = NetworkType{:resnet,1}()
(Base.String(::NetworkType{T}) where T) = String(T)

function getName(network_type::NetworkType{T,D}) where {T,D}
    name = String(network_type)
    if (T == :spline)
        name = name*string(D)
    end
    return name
end

# glorot_uniform_tensor(T, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0) / sum(dims[1:end-1]))
function glorot_uniform_tensor(T, dims...)
    Random.seed!(0)
    return (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0) / sum(dims[1:end-1]))
end

## Spline parameters
Base.@kwdef mutable struct SplineParameters{T,D,F}
    width::Int  = 2                 # number of neurons per layer
    nKnots::Int = 4                 # number of knots
    σ::F = relu                     # activation function
    hLayers::T = 0.05               # layer-to-layer step size
    nLayers::Int = 20               # number of layers
    tEnd::T = hLayers*nLayers       # end of time interval
    δKnots::T  = tEnd/nKnots        # spline grid size
    amplitude::T = 1.0              # initialization amplitude
end
# default activation set to relu
(SplineParameters{T,D}(;σ=relu, kwargs...) where {T,D}) =
    SplineParameters{T,D,typeof(σ)}(;σ=σ, kwargs...)
# default degree set to 2
(SplineParameters{T}(;σ=relu,kwargs...) where T) = SplineParameters{T,2}(;σ=σ, kwargs...)

degree(::SplineParameters{T,D}) where {T,D} = D

##  Construct the whole network
#=   T: data type, usually Float32 or Float64
     F: activation fucntion type, relu, tanh ...
     D: degree of the spline
=#
mutable struct SplineNetwork{T,D,F}
    args::SplineParameters{T,D,F}
    # all the parameters in network
    cfWeights::Array{T,3}   # coefficients of spline bases of weights
    cfBiases::Array{T,2}    # coefficients of spline bases of biases
    # gradients of the parameters
    gdWeights::Array{T,3}   # gradients of spline bases of weights
    gdBiases::Array{T,2}    # gradients of spline bases of biases
    # workspace used for local weights & biases
    K::Array{T,2}
    b::Array{T,1}
    batchSize::Int
    # workspace used for computations
    xInterm::Array{T,2}
    vInterm::Array{T,2}
    zInterm::Array{T,2}
    sInterm::Array{T,2}
    # storage of all the state variables
    states::Array{T,3}
    # non-zero b-spline coefficients vector
    bsp::BSpline{T,D}
end

function SplineNetwork(args::SplineParameters{T,D,F}; batchSize=1, init_weights_zeros=false, init_biases_zeros=false) where {T,D,F}
    nKnots = args.nKnots
    width = args.width
    nLayers = args.nLayers
    amplitude = args.amplitude
    # initialization
    return SplineNetwork{T,D,F}(
        args,
        # all the parameters in network
        init_weights_zeros ? zeros(T, width, width, nKnots+D) : amplitude .* glorot_uniform_tensor(T, width, width, nKnots+D),
        init_biases_zeros ? zeros(T, width, nKnots+D) : amplitude .* glorot_uniform_tensor(T, width, nKnots+D),
        # gradients of the parameters
        zeros(T, width, width, nKnots+D),
        zeros(T, width, nKnots+D),
        # workspace used for local weights & biases
        zeros(T, width, width),
        zeros(T, width),
        # batch size, default to 1
        batchSize,
        # workspace used for computations
        zeros(T, width, batchSize),
        zeros(T, width, batchSize),
        zeros(T, width, batchSize),
        zeros(T, width, batchSize),
        # states stored
        zeros(T, width, batchSize, nLayers+1),
        # non-zero b-spline coefficients vector
        BSpline{T,D}(),
    )
end

function set_batch_size(spline_net::SplineNetwork{T,D,F}, batchSize) where {T,D,F}
    # if batch size hasn't changed, do nothing
    if spline_net.batchSize == batchSize
        return
    end
    width = spline_net.args.width
    nKnots = spline_net.args.nKnots
    nLayers = spline_net.args.nLayers
    spline_net.batchSize = batchSize
    spline_net.xInterm = zeros(T, width, batchSize)
    spline_net.vInterm = zeros(T, width, batchSize)
    spline_net.zInterm = zeros(T, width, batchSize)
    spline_net.sInterm = zeros(T, width, batchSize)
    spline_net.states = zeros(T, width, batchSize, nLayers+1)
end

## Layer-to-layer propagation
function spline_layer!(spline_net::SplineNetwork{T,D,F}, state, t, l) where {T,D,F}
    K = spline_net.K
    b = spline_net.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_net.args.δKnots

    bspline_coefficients_update!(spline_net.bsp, l, t, δ)

    for k in 1:(D+1)
        wslice = uview(spline_net.cfWeights,:,:,l+k)
        bslice = uview(spline_net.cfBiases,:,l+k)

        K .+= spline_net.bsp.Bx[k] .* wslice
        b .+= spline_net.bsp.Bx[k] .* bslice
    end

    h = spline_net.args.hLayers
    xInterm = spline_net.xInterm

    # if the batch size is 1, use matvec instead of matmat for better performance
    if spline_net.batchSize == 1
        mul!(uview(xInterm,:,1), K, uview(state, :, 1))
    else
        mul!(xInterm, K, state)
    end

    state .+= h .* spline_net.args.σ.(xInterm .+ b)
end
## Forward pass
function apply_network(spline_net::SplineNetwork, x::AbstractMatrix)
    batchSize = size(x,2)
    set_batch_size(spline_net, batchSize)

    args = spline_net.args
    width = args.width
    # store the first state variable
    spline_net.states[:,:,1] = x
    state = spline_net.sInterm
    # make a copy of input rather than mutating it
    state .= x
    # loop to get all the state variables
    for i in 1:args.nLayers
        t = (i - 1)*args.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/args.δKnots)         # t ϵ [lδ, (l+1)δ]
        spline_layer!(spline_net, state, t, l)
        spline_net.states[:,:,i+1] = state
    end
    return spline_net.sInterm
end

function apply_network(spline_net::SplineNetwork, x::AbstractVector)
    x_matrix = reshape(x, size(x,1), 1)
    apply_network(spline_net, x_matrix)
end

## Use the previous adjoint to update gradients at ilayer
function gradient_update_local!(spline_net::SplineNetwork{T,D,F}, t, l, ilayer) where {T,D,F}
    K = spline_net.K
    b = spline_net.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_net.args.δKnots

    bspline_coefficients_update!(spline_net.bsp, l, t, δ)
    for k in 1:D+1
        wslice = uview(spline_net.cfWeights,:,:,l+k)
        bslice = uview(spline_net.cfBiases,:,l+k)

        K .+= spline_net.bsp.Bx[k] .* wslice
        b .+= spline_net.bsp.Bx[k] .* bslice
    end

    h = spline_net.args.hLayers
    # state variable u at ilayer
    ucurrent = uview(spline_net.states,:,:,ilayer)
    # zInterm:  intermediate variable σ'(K_n u_n + b_n) * v_n+1
    # vInterm:  the adjoint of next layer
    zInterm = spline_net.zInterm

    if spline_net.batchSize == 1
        mul!(uview(zInterm,:,1), K, uview(ucurrent,:,1))
    else
        mul!(zInterm, K, ucurrent)
    end
    zInterm .+= b
    zInterm .= spline_net.args.σ'.(zInterm) .* spline_net.vInterm
    # For loop, entrywise in-place assignment
    width = spline_net.args.width
    for b in 1:spline_net.batchSize
        for i in 1:width
            for j in 1:width
                for k in 1:D+1
                    spline_net.gdWeights[i,j,l+k] += h * spline_net.bsp.Bx[k] * zInterm[i,b] * ucurrent[j,b]
                end
            end
            for k in 1:length(spline_net.bsp.Bx)
                spline_net.gdBiases[i,l+k] += h * spline_net.bsp.Bx[k] * zInterm[i,b]
            end
        end
    end
    # Update adjoint after updating the gradients!
    xInterm = spline_net.xInterm
    if spline_net.batchSize == 1
        mul!(uview(xInterm,:,1), K', uview(zInterm,:,1))
    else
        mul!(xInterm, K', zInterm)
    end
    spline_net.vInterm .+= h .* xInterm
end
## a function to update all the gradients of SplineNetwork
function gradient_update!(spline_net::SplineNetwork{T,D,F}, z::AbstractMatrix) where {T,D,F}
    args = spline_net.args
    spline_net.vInterm .= z    # last adjoint
    # set gradient tensors to 0
    fill!(spline_net.gdWeights, T(0.0))
    fill!(spline_net.gdBiases, T(0.0))
    # loop in reverse order
    for i in args.nLayers:-1:1
        t = (i - 1)*args.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/args.δKnots)           # t ϵ [lδ, (l+1)δ]
        gradient_update_local!(spline_net, t, l, i)
    end
end

function gradient_update!(spline_net::SplineNetwork, z::AbstractVector)
    z_matrix = reshape(z, size(z,1), 1)
    gradient_update!(spline_net, z_matrix)
end
## custom adjoint
function pullback_network(cx, spline_net::SplineNetwork, x)
    y = apply_network(spline_net::SplineNetwork, x)
    back = function(z)
        gradient_update!(spline_net, z)
        Zygote.accum_param(cx, spline_net.cfWeights, spline_net.gdWeights)
        Zygote.accum_param(cx, spline_net.cfBiases, spline_net.gdBiases)
        return ((cfWeights=spline_net.gdWeights, cfBiases=spline_net.gdBiases), spline_net.vInterm)
    end
    return y, back
end
## Tell Zygote about our custom adjoint
Zygote.@adjoint apply_network(spline_net::SplineNetwork, x) = pullback_network(__context__, spline_net, x)

## Constructors & trainable parameters
(spline_net::SplineNetwork)(x) = apply_network(spline_net, x)
Flux.trainable(spline_net::SplineNetwork) = (spline_net.cfWeights, spline_net.cfBiases,)
