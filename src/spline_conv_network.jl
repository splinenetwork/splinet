using Flux, Zygote
using Base: @kwdef
using BenchmarkTools
using LinearAlgebra
using UnsafeArrays

include("bsplines.jl")
glorot_uniform_tensor(T, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0) / sum(dims[1:end-1]))

## Convolution Layers parameters
Base.@kwdef mutable struct ConvParameters
    filterDim::Int = 5              # kernel dimension
    stride::Int = 1
    # padding style is equally on both ends of each dimension
    # this might require filter dimension to be odd
    pad::Int = floor(Int, filterDim/2)
end
## Spline parameters
Base.@kwdef mutable struct SplineParameters{T,D,F}
    width::Int  = 100               # number of neurons per layer
    nKnots::Int = 5                 # number of knots
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
mutable struct SplineConvNetwork{T,D,F}
    convArgs::ConvParameters
    args::SplineParameters{T,D,F}
    # all the parameters in network
    cfWeights::Array{T,2}   # coefficients of spline bases of weights
    cfBiases::Array{T,2}    # coefficients of spline bases of biases
    # gradients of the parameters
    gdWeights::Array{T,2}   # gradients of spline bases of weights
    gdBiases::Array{T,2}    # gradients of spline bases of biases
    # workspace used for local weights & biases
    K::Array{T,3}
    b::Array{T,1}
    batchSize::Int
    # workspace used for computations
    xInterm::Array{T,3}
    vInterm::Array{T,3}
    zInterm::Array{T,3}
    sInterm::Array{T,3}
    zConVu::Array{T,3}
    # storage of all the state variables
    states::Array{T,3}
    # non-zero b-spline coefficients vector
    bsp::BSpline{T,D}
end

function SplineConvNetwork(convArgs::ConvParameters, args::SplineParameters{T,D,F}; batchSize=10, init_weights_zeros=false, init_biases_zeros=false) where {T,D,F}
    filterDim = convArgs.filterDim
    nKnots = args.nKnots
    width = args.width
    nLayers = args.nLayers
    amplitude = args.amplitude
    # initialization
    return SplineConvNetwork{T,D,F}(
        convArgs,
        args,
        # all the parameters in network
        init_weights_zeros ? zeros(T, filterDim, nKnots+D) : amplitude .* glorot_uniform_tensor(T, filterDim, nKnots+D),
        init_biases_zeros ? zeros(T, width, nKnots+D) : amplitude .* glorot_uniform_tensor(T, width, nKnots+D),
        # gradients of the parameters
        zeros(T, filterDim, nKnots+D),
        zeros(T, width, nKnots+D),
        # workspace used for local weights & biases
        zeros(T, filterDim, 1, 1),
        zeros(T, width),
        # batch size, default to 1
        batchSize,
        # workspace used for computations
        zeros(T, width, 1, batchSize),
        zeros(T, width, 1, batchSize),
        zeros(T, width, 1, batchSize),
        zeros(T, width, 1, batchSize),
        # zConVu should have the same dimension as K
        zeros(T, filterDim, 1, 1),
        # states stored
        zeros(T, width, batchSize, nLayers+1),
        # non-zero b-spline coefficients vector
        BSpline{T,D}(),
    )
end

function set_batch_size(spline_conv_net::SplineConvNetwork{T,D,F}, batchSize) where {T,D,F}
    # if batch size hasn't changed, do nothing
    if spline_conv_net.batchSize == batchSize
        return
    end
    width = spline_conv_net.args.width
    nKnots = spline_conv_net.args.nKnots
    nLayers = spline_conv_net.args.nLayers
    spline_conv_net.batchSize = batchSize
    spline_conv_net.xInterm = zeros(T, width, 1, batchSize)
    spline_conv_net.vInterm = zeros(T, width, 1, batchSize)
    spline_conv_net.zInterm = zeros(T, width, 1, batchSize)
    spline_conv_net.sInterm = zeros(T, width, 1, batchSize)
    spline_conv_net.states = zeros(T, width, batchSize, nLayers+1)
end
## Layer-to-layer propagation
function spline_conv_layer!(spline_conv_net::SplineConvNetwork{T,D,F}, state, t, l) where {T,D,F}
    K = spline_conv_net.K
    b = spline_conv_net.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_conv_net.args.δKnots

    bspline_coefficients_update!(spline_conv_net.bsp, l, t, δ)
    for k in 1:(D+1)
        wslice = uview(spline_conv_net.cfWeights,:,l+k)
        bslice = uview(spline_conv_net.cfBiases,:,l+k)

        for i in 1:spline_conv_net.convArgs.filterDim
            K[i,1,1] += spline_conv_net.bsp.Bx[k] * wslice[i]
        end
        b .+= spline_conv_net.bsp.Bx[k] .* bslice
    end

    h = spline_conv_net.args.hLayers
    xInterm = spline_conv_net.xInterm

    ######################################################################
    # This is the main part which is different from dense layer SpliNet
    # A convolution is implemented instead
    pad = spline_conv_net.convArgs.pad
    conv!(xInterm, state, K, DenseConvDims(size(state), size(K); stride=1, padding=pad))
    ######################################################################

    for bi in 1:spline_conv_net.batchSize
        for i in 1:spline_conv_net.args.width
            state[i,1,bi] += h * spline_conv_net.args.σ(xInterm[i,1,bi] + b[i])
        end
    end
end
## Forward pass
function apply_network(spline_conv_net::SplineConvNetwork, x::AbstractMatrix)
    batchSize = size(x,2)
    set_batch_size(spline_conv_net, batchSize)

    args = spline_conv_net.args
    width = args.width
    # store the first state variable
    spline_conv_net.states[:,:,1] = x
    state = spline_conv_net.sInterm
    # make a copy of input rather than mutating it
    state[:,1,:] = x
    # loop to get all the state variables
    for i in 1:args.nLayers
        t = (i - 1)*args.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/args.δKnots)           # t ϵ [lδ, (l+1)δ]
        spline_conv_layer!(spline_conv_net, state, t, l)
        sview = uview(state,:,1,:)
        spline_conv_net.states[:,:,i+1] = sview
    end
    return spline_conv_net.sInterm[:,1,:]
end

function apply_network(spline_conv_net::SplineConvNetwork, x::AbstractVector)
    x_matrix = reshape(x, size(x,1), 1)
    apply_network(spline_conv_net, x_matrix)
end
## Use the previous adjoint to update gradients at ilayer
function gradient_update_local!(spline_conv_net::SplineConvNetwork{T,D,F}, t, l, ilayer) where {T,D,F}
    K = spline_conv_net.K
    b = spline_conv_net.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_conv_net.args.δKnots

    bspline_coefficients_update!(spline_conv_net.bsp, l, t, δ)
    for k in 1:D+1
        wslice = uview(spline_conv_net.cfWeights,:,l+k)
        bslice = uview(spline_conv_net.cfBiases,:,l+k)

        for i in 1:spline_conv_net.convArgs.filterDim
            K[i,1,1] += spline_conv_net.bsp.Bx[k] * wslice[i]
        end
        b .+= spline_conv_net.bsp.Bx[k] .* bslice
    end

    h = spline_conv_net.args.hLayers
    # state variable u at ilayer, batched
    ucurrent = uview(spline_conv_net.states,:,:,ilayer)
    # zInterm:  intermediate variable σ'(K_n u_n + b_n) * v_n+1
    # vInterm:  the adjoint of next layer
    zInterm = spline_conv_net.zInterm
    xInterm = spline_conv_net.xInterm
    xInterm[:,1,:] = ucurrent

    ######################################################################
    pad = spline_conv_net.convArgs.pad
    cdims = DenseConvDims(size(xInterm), size(K); stride=1, padding=pad)
    conv!(zInterm, xInterm, K, cdims)
    ######################################################################

    zInterm .+= b
    zInterm .= spline_conv_net.args.σ'.(zInterm) .* spline_conv_net.vInterm

    # For loop, entrywise in-place assignment
    zConVu = spline_conv_net.zConVu
    ######################################################################
    ∇conv_filter!(zConVu, xInterm, zInterm, cdims)
    ######################################################################

    for k in 1:length(spline_conv_net.bsp.Bx)
        for i in 1:spline_conv_net.convArgs.filterDim
            spline_conv_net.gdWeights[i,l+k] += h * spline_conv_net.bsp.Bx[k] * zConVu[i,1,1]
        end
    end

    for bi in 1:spline_conv_net.batchSize
        for k in 1:length(spline_conv_net.bsp.Bx)
            for i in 1:spline_conv_net.args.width
                spline_conv_net.gdBiases[i,l+k] += h * spline_conv_net.bsp.Bx[k] * zInterm[i,1,bi]
            end
        end
    end
    # Update adjoint after updating the gradients!
    ######################################################################
    ∇conv_data!(xInterm, zInterm, K, cdims)
    ######################################################################
    spline_conv_net.vInterm .+= h .* xInterm
end
## a function to update all the gradients of SplineConvNetwork
function gradient_update!(spline_conv_net::SplineConvNetwork{T,D,F}, z::AbstractMatrix) where {T,D,F}
    args = spline_conv_net.args
    spline_conv_net.vInterm[:,1,:] = z    # last adjoint
    # set gradient tensors to 0
    fill!(spline_conv_net.gdWeights, T(0.0))
    fill!(spline_conv_net.gdBiases, T(0.0))
    # loop in reverse order
    for i in args.nLayers:-1:1
        t = (i - 1)*args.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/args.δKnots)           # t ϵ [lδ, (l+1)δ]
        gradient_update_local!(spline_conv_net, t, l, i)
    end
end

function gradient_update!(spline_conv_net::SplineConvNetwork, z::AbstractVector)
    z_matrix = reshape(z, size(z,1), 1)
    gradient_update!(spline_conv_net, z_matrix)
end
## custom adjoint
function pullback_network(cx, spline_conv_net::SplineConvNetwork, x)
    y = apply_network(spline_conv_net::SplineConvNetwork, x)
    back = function(z)
        gradient_update!(spline_conv_net, z)
        Zygote.accum_param(cx, spline_conv_net.cfWeights, spline_conv_net.gdWeights)
        Zygote.accum_param(cx, spline_conv_net.cfBiases, spline_conv_net.gdBiases)
        return ((cfWeights=spline_conv_net.gdWeights, cfBiases=spline_conv_net.gdBiases), spline_conv_net.vInterm[:,1,:])
    end
    return y, back
end
## Tell Zygote about our custom adjoint
Zygote.@adjoint apply_network(spline_conv_net::SplineConvNetwork, x) = pullback_network(__context__, spline_conv_net, x)

## Constructors & trainable parameters
(spline_conv_net::SplineConvNetwork)(x) = apply_network(spline_conv_net, x)
Flux.trainable(spline_conv_net::SplineConvNetwork) = (spline_conv_net.cfWeights, spline_conv_net.cfBiases,)
