using Flux, Zygote, BenchmarkTools, LinearAlgebra, UnsafeArrays
using Base: @kwdef

include("bsplines.jl")
#
glorot_uniform_tensor(T, dims...) = (rand(T, dims...) .- T(0.5)) .* sqrt(T(24.0) / sum(dims[1:end-1]))

## Convolution layers parameters
Base.@kwdef mutable struct ConvParameters
    filterDims::Tuple = (1,1,1)          # kernel dimension
    stride::Int = 1
    # padding style is equally on both ends of each dimension
    # this might require filter dimension to be odd
    pad::Tuple = floor.(Int, filterDims./2)
    chan_in::Int = 1
    chan_out::Int = 1
    cdims::Tuple = (filterDims..., chan_in, chan_out)
end
## Network parameters
#=   T: data type, usually Float32 or Float64
     F: activation fucntion type, relu, tanh ...
     D: degree of the spline
=#
Base.@kwdef mutable struct SplineParameters{T,D,F}
    # data dimension: length, width, depth, channel
    nKnots::Int = 10
    σ::F = relu
    hLayers::T = 0.05               # layer-to-layer step size
    nLayers::Int = 20               # number of layers
    tEnd::T = hLayers*nLayers       # end of time interval
    δKnots::T  = tEnd/nKnots        # spline grid size
    amplitude::T = 1.0              # initialization amplitude
end
# default activation set to relu
(SplineParameters{T,D}(;kwargs...) where {T,D}) = SplineParameters{T,D,typeof(relu)}(;kwargs...)
# default degree set to 2
(SplineParameters{T}(;kwargs...) where T) = SplineParameters{T,2,typeof(relu)}(;kwargs...)
# degree get function
degree(::SplineParameters{T,D}) where {T,D} = D

mutable struct DenseNetworkBackend{T,D,F}
    # Dense network only takes a vector as input
    # hence dimension only need a integer
    width::Int
    spArgs::SplineParameters{T,D,F}
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
    vInterm::Array{T,2}     # adjoints
    zInterm::Array{T,2}
    sInterm::Array{T,2}     # for state storage
    tInterm::Array{T,2}     # for λ
    # increment of gradient of K from one spline basis
    Kgrad::Array{T,2}
    # storage of all the state variables
    states::Array{T,3}
    # time scalar
    λ::Array{T,1}
    dλ::Array{T,1}
end

function DenseNetworkBackend(width::Int, spArgs::SplineParameters{T,D,F};
            batchSize=10, init_λ=1.0, init_weights_zeros=false, init_biases_zeros=false) where{T,D,F}
    nKnots = spArgs.nKnots
    nLayers = spArgs.nLayers
    amplitude = spArgs.amplitude
    return DenseNetworkBackend{T,D,F}(
        width,
        spArgs,
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
        zeros(T, width, batchSize),
        # Kgrad should have the same dimension as K
        zeros(T, width, width),
        # states stored
        zeros(T, width, batchSize, nLayers+1),
        [init_λ],
        [0.0],
    )
end

mutable struct DenseAntisymmetricNetworkBackend{T,D,F}
    # Dense network only takes a vector as input
    # hence dimension only need a integer
    width::Int
    spArgs::SplineParameters{T,D,F}
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
    vInterm::Array{T,2}     # adjoints
    zInterm::Array{T,2}
    sInterm::Array{T,2}     # for state storage
    tInterm::Array{T,2}     # for λ
    # increment of gradient of K from one spline basis
    Kgrad::Array{T,2}
    # storage of all the state variables
    states::Array{T,3}
    # time scalar
    λ::Array{T,1}
    dλ::Array{T,1}
    # eigenvalue shift
    γ::T
end

function DenseAntisymmetricNetworkBackend(width::Int, spArgs::SplineParameters{T,D,F};
            batchSize=10, init_λ=1.0, γ=0.3, init_weights_zeros=false, init_biases_zeros=false) where{T,D,F}
    nKnots = spArgs.nKnots
    nLayers = spArgs.nLayers
    amplitude = spArgs.amplitude
    return DenseAntisymmetricNetworkBackend{T,D,F}(
        width,
        spArgs,
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
        zeros(T, width, batchSize),
        # Kgrad should have the same dimension as K
        zeros(T, width, width),
        # states stored
        zeros(T, width, batchSize, nLayers+1),
        [init_λ],
        [0.0],
        γ
    )
end

mutable struct ConvNetworkBackend{T,D,F}
    dataDims::NTuple{4,Int}
    spArgs::SplineParameters{T,D,F}
    convArgs::ConvParameters
    # all the parameters in network
    cfWeights::Array{T,6}   # coefficients of spline bases of weights
    cfBiases::Array{T,5}    # coefficients of spline bases of biases
    # gradients of the parameters
    gdWeights::Array{T,6}   # gradients of spline bases of weights
    gdBiases::Array{T,5}    # gradients of spline bases of biases
    # workspace used for local weights & biases
    K::Array{T,5}
    b::Array{T,4}
    batchSize::Int
    # workspace used for computations
    xInterm::Array{T,5}
    vInterm::Array{T,5}     # adjoints
    zInterm::Array{T,5}
    sInterm::Array{T,5}     # for state storage
    tInterm::Array{T,5}     # for λ
    # increment of gradient of K from one spline basis
    Kgrad::Array{T,5}
    # storage of all the state variables
    states::Array{T,6}
    # time scalar
    λ::Array{T,1}
    dλ::Array{T,1}
end

function ConvNetworkBackend(spArgs::SplineParameters{T,D,F}, convArgs::ConvParameters; dataDims=(1,1,1,1),
            batchSize=10, init_λ=1.0, init_weights_zeros=false, init_biases_zeros=false) where{T,D,F}
    nKnots = spArgs.nKnots
    nLayers = spArgs.nLayers
    amplitude = spArgs.amplitude
    cdims = convArgs.cdims
    return ConvNetworkBackend{T,D,F}(
        dataDims,
        spArgs,
        convArgs,
        # all the parameters in network
        init_weights_zeros ? zeros(T, cdims..., nKnots+D) : amplitude .* glorot_uniform_tensor(T, cdims..., nKnots+D),
        init_biases_zeros ? zeros(T, dataDims..., nKnots+D) : amplitude .* glorot_uniform_tensor(T, dataDims..., nKnots+D),
        # gradients of the parameters
        zeros(T, cdims..., nKnots+D),
        zeros(T, dataDims..., nKnots+D),
        # workspace used for local weights & biases
        zeros(T, cdims...),
        zeros(T, dataDims...),
        # batch size, default to 1
        batchSize,
        # workspace used for computations
        zeros(T, dataDims..., batchSize),
        zeros(T, dataDims..., batchSize),
        zeros(T, dataDims..., batchSize),
        zeros(T, dataDims..., batchSize),
        zeros(T, dataDims..., batchSize),
        # Kgrad should have the same dimension as K
        zeros(T, cdims...),
        # states stored
        zeros(T, dataDims..., batchSize, nLayers+1),
        [init_λ],
        [0.0],
    )
end

mutable struct SplineNetwork{T,D,F,NetworkType}
    spArgs::SplineParameters{T,D,F}
    batchSize::Int
    bsp::BSpline{T,D}
    networkbackend::NetworkType
end

function SplineNetwork(args::SplineParameters{T,D,F}, data::NetworkType) where {T,D,F,NetworkType}
    # initialization
    return SplineNetwork{T,D,F,NetworkType}(
        args,
        data.batchSize,
        BSpline{T,D}(),
        data,
    )
end

# change batch size if needed
function set_batch_size(nn::Union{DenseNetworkBackend{T,D,F},DenseAntisymmetricNetworkBackend{T,D,F}}, batchSize) where {T,D,F}
    # if batch size hasn't changed, do nothing
    if nn.batchSize == batchSize
        return
    end
    spArgs = nn.spArgs
    width = nn.width
    nLayers = spArgs.nLayers
    nn.batchSize = batchSize
    nn.xInterm = zeros(T, width, batchSize)
    nn.vInterm = zeros(T, width, batchSize)
    nn.zInterm = zeros(T, width, batchSize)
    nn.sInterm = zeros(T, width, batchSize)
    nn.tInterm = zeros(T, width, batchSize)
    nn.states = zeros(T, width, batchSize, nLayers+1)
end

function set_batch_size(nn::ConvNetworkBackend{T,D,F}, batchSize) where {T,D,F}
    # if batch size hasn't changed, do nothing
    if nn.batchSize == batchSize
        return
    end
    spArgs = nn.spArgs
    dataDims = nn.dataDims
    nLayers = spArgs.nLayers
    nn.batchSize = batchSize
    nn.xInterm = zeros(T, dataDims..., batchSize)
    nn.vInterm = zeros(T, dataDims..., batchSize)
    nn.zInterm = zeros(T, dataDims..., batchSize)
    nn.sInterm = zeros(T, dataDims..., batchSize)
    nn.tInterm = zeros(T, dataDims..., batchSize)
    nn.states = zeros(T, dataDims..., batchSize, nLayers+1)
end

get_matrix(::DenseNetworkBackend, K::AbstractArray) = K
get_matrix(nn::DenseAntisymmetricNetworkBackend, K::AbstractArray) = K - K' - nn.γ*I

function apply_linear!(nn::DenseNetworkBackend, K::AbstractArray,
            x::AbstractArray, output::AbstractArray)
    mul!(output, K, x)
end

function apply_linear!(nn::DenseAntisymmetricNetworkBackend, K::AbstractArray,
    x::AbstractArray, output::AbstractArray)
    mul!(output, K, x)
    mul!(output, K', x, 1.0, 0.0)
    output += nn.γ*x
end

function apply_linear!(nn::ConvNetworkBackend, K::AbstractArray,
            x::AbstractArray, output::AbstractArray)
    cdims = DenseConvDims(size(x), size(K); stride=nn.convArgs.stride, padding=nn.convArgs.pad)
    conv!(output, x, K, DenseConvDims(size(x), size(K); stride=nn.convArgs.stride, padding=nn.convArgs.pad))
end

## obtain zero-alloctated dimension tuple of A
@generated function colon_tuple(::Val{D}) where {D}
    c = Tuple(Colon() for j=1:D-1)
    # :($c)
end
function separate_slice(A::AbstractArray{T,D}, i) where {T,D}
    uview(A, colon_tuple(Val(D))..., i)
end

## Layer-to-layer propagation
function spline_layer!(spline_net::SplineNetwork{T,D,F,NetworkType}, state, t, l) where {T,D,F,NetworkType}
    networkbackend = spline_net.networkbackend
    K = networkbackend.K
    b = networkbackend.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_net.spArgs.δKnots

    bspline_coefficients_update!(spline_net.bsp, l, t, δ)
    for k in 1:(D+1)
        # wslice = uview(spline_net.cfWeights,:,:,:,:,:,l+k)
        wslice = separate_slice(networkbackend.cfWeights, l+k)
        # bslice = uview(spline_net.cfBiases,:,:,:,:,l+k)
        bslice = separate_slice(networkbackend.cfBiases, l+k)

        K .+= spline_net.bsp.Bx[k] .* wslice
        b .+= spline_net.bsp.Bx[k] .* bslice
    end

    h = spline_net.spArgs.hLayers
    λ = networkbackend.λ
    xInterm = networkbackend.xInterm
    ######################################################################
    #  depending on network type to apply different linear transformation
    apply_linear!(networkbackend, K, state, xInterm)
    ######################################################################

    state .+= (h * λ[1]) .* spline_net.spArgs.σ.(xInterm .+ b)
end
## Forward pass
function apply_network(spline_net::SplineNetwork, x::AbstractArray)
    batchSize = size(x)[end]
    set_batch_size(spline_net.networkbackend, batchSize)

    spArgs = spline_net.spArgs
    # store the first state variable
    input = separate_slice(spline_net.networkbackend.states, 1)
    # spline_net.networkbackend.states[:,:,:,:,:,1] = x
    input .= x

    # state is a pointer to workspace sInterm
    state = spline_net.networkbackend.sInterm
    state .= x      # x is not changed in this function
    # loop to get all the state variables
    for i in 1:spArgs.nLayers
        t = (i - 1)*spArgs.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/spArgs.δKnots)           # t ϵ [lδ, (l+1)δ]
        spline_layer!(spline_net, state, t, l)
        # sview = uview(state,:,:,:,:,:)
        istate = separate_slice(spline_net.networkbackend.states, i+1)
        istate .= state
        # spline_net.networkbackend.states[:,:,:,:,:,i+1] = state
    end
    return spline_net.networkbackend.sInterm
end
## customized gradient functions
function ∇linear_transformation_filter!(nn::DenseNetworkBackend, xInterm::AbstractArray,
                zInterm::AbstractArray, Kgrad::AbstractArray)
    mul!(Kgrad, zInterm, xInterm')
end

function ∇linear_transformation_filter!(nn::DenseAntisymmetricNetworkBackend, xInterm::AbstractArray,
    zInterm::AbstractArray, Kgrad::AbstractArray)
    mul!(Kgrad, zInterm, xInterm')
    mul!(Kgrad, xInterm, zInterm', 1.0, 0.0)
end

function ∇linear_transformation_filter!(nn::ConvNetworkBackend, xInterm::AbstractArray,
                zInterm::AbstractArray, Kgrad::AbstractArray)
    stride = nn.convArgs.stride
    pad = nn.convArgs.pad
    cdims = DenseConvDims(size(xInterm), size(Kgrad); stride=stride, padding=pad)
    ∇conv_filter!(Kgrad, xInterm, zInterm, cdims)
end

function ∇linear_transformation_data!(nn::DenseNetworkBackend, zInterm::AbstractArray,
                K::AbstractArray, xInterm::AbstractArray)
    mul!(xInterm, K', zInterm)
end

function ∇linear_transformation_data!(nn::DenseAntisymmetricNetworkBackend, zInterm::AbstractArray,
    K::AbstractArray, xInterm::AbstractArray)
    mul!(xInterm, K', zInterm)
    mul!(xInterm, K, zInterm, 1.0, 0.0)
    xInterm += nn.γ*zInterm
end

function ∇linear_transformation_data!(nn::ConvNetworkBackend, zInterm::AbstractArray,
                K::AbstractArray, xInterm::AbstractArray)
    stride = nn.convArgs.stride
    pad = nn.convArgs.pad
    cdims = DenseConvDims(size(xInterm), size(K); stride=stride, padding=pad)
    ∇conv_data!(xInterm, zInterm, K, cdims)
end
## Use the previous adjoint to update gradients at ilayer
function gradient_update_local!(spline_net::SplineNetwork{T,D,F,NetworkType}, t, l, ilayer) where {T,D,F,NetworkType}
    networkbackend = spline_net.networkbackend
    K = networkbackend.K
    b = networkbackend.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_net.spArgs.δKnots

    bspline_coefficients_update!(spline_net.bsp, l, t, δ)
    for k in 1:(D+1)
        # wslice = uview(spline_net.cfWeights,:,:,:,:,:,l+k)
        wslice = separate_slice(networkbackend.cfWeights, l+k)
        # bslice = uview(spline_net.cfBiases,:,:,:,:,l+k)
        bslice = separate_slice(networkbackend.cfBiases, l+k)

        K .+= spline_net.bsp.Bx[k] .* wslice
        b .+= spline_net.bsp.Bx[k] .* bslice
    end

    h = spline_net.spArgs.hLayers
    λ = networkbackend.λ
    # zInterm:  intermediate variable σ'(K_n * u_n + b_n) * v_n+1
    # vInterm:  the adjoint of next layer
    # xInterm:  state variable u at ilayer, batched
    #           later be used for storage of intermediate variable
    # tInterm:  intermediate variable h * σ(K_n * u_n + b_n) * v_n+1
    zInterm = networkbackend.zInterm
    xInterm = networkbackend.xInterm
    tInterm = networkbackend.tInterm
    xInterm .= separate_slice(networkbackend.states, ilayer)

    ######################################################################
    #  depending on network type to apply different linear transformation
    apply_linear!(networkbackend, K, xInterm, zInterm)
    ######################################################################

    # The rest computation of intermediate variable z is the same
    # for dense and convoluation layers
    zInterm .+= b
    tInterm .= h .* spline_net.spArgs.σ.(zInterm) .* networkbackend.vInterm
    zInterm .= spline_net.spArgs.σ'.(zInterm) .* networkbackend.vInterm

    # For loop, entrywise in-place assignment
    Kgrad = networkbackend.Kgrad
    ######################################################################
    #  depending on network type to apply different linear transformation
    ∇linear_transformation_filter!(networkbackend, xInterm, zInterm, Kgrad)
    ######################################################################

    # Loop over the spline bases & batch size to accumulate the gradient
    for k in 1:length(spline_net.bsp.Bx)
        separate_slice(networkbackend.gdWeights, l+k) .+= (h * λ[1]) .* spline_net.bsp.Bx[k] .* Kgrad
    end

    for bi in 1:spline_net.batchSize
        zSingle = separate_slice(zInterm, bi)
        for k in 1:length(spline_net.bsp.Bx)
            separate_slice(networkbackend.gdBiases, l+k) .+= (h * λ[1]) .* spline_net.bsp.Bx[k] .* zSingle
        end
    end
    networkbackend.dλ[1] += sum(tInterm)

    # Update adjoint after updating the gradients!
    ######################################################################
    #  depending on network type to apply different linear transformation
    ∇linear_transformation_data!(networkbackend, zInterm, K, xInterm)
    ######################################################################
    networkbackend.vInterm .+= (h * λ[1]) .* xInterm
end

## a function to update all the gradients of SplineNetwork
function gradient_update!(spline_net::SplineNetwork{T,D,F,NetworkType}, z::AbstractArray) where {T,D,F,NetworkType}
    spArgs = spline_net.spArgs
    spline_net.networkbackend.vInterm .= z    # last adjoint
    # set gradient tensors to 0
    fill!(spline_net.networkbackend.gdWeights, T(0.0))
    fill!(spline_net.networkbackend.gdBiases, T(0.0))
    fill!(spline_net.networkbackend.dλ, T(0.0))
    # loop in reverse order
    for i in spArgs.nLayers:-1:1
        t = (i - 1)*spArgs.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/spArgs.δKnots)           # t ϵ [lδ, (l+1)δ]
        gradient_update_local!(spline_net, t, l, i)
    end
end

## layer spectrum check
function layer_spectrum_check(spline_net::SplineNetwork{T,D,F,NetworkType}, state, t, l) where {T,D,F,NetworkType}
    networkbackend = spline_net.networkbackend
    K = networkbackend.K
    b = networkbackend.b
    fill!(K, T(0.0))
    fill!(b, T(0.0))
    δ = spline_net.spArgs.δKnots

    bspline_coefficients_update!(spline_net.bsp, l, t, δ)
    for k in 1:(D+1)
     # wslice = uview(spline_net.cfWeights,:,:,:,:,:,l+k)
     wslice = separate_slice(networkbackend.cfWeights, l+k)
     # bslice = uview(spline_net.cfBiases,:,:,:,:,l+k)
     bslice = separate_slice(networkbackend.cfBiases, l+k)

     K .+= spline_net.bsp.Bx[k] .* wslice
     b .+= spline_net.bsp.Bx[k] .* bslice
    end

    h = spline_net.spArgs.hLayers
    λ = networkbackend.λ
    xInterm = networkbackend.xInterm
    ######################################################################
    #  depending on network type to apply different linear transformation
    apply_linear!(networkbackend, K, state, xInterm)
    W = get_matrix(networkbackend, K)
    ######################################################################

    σ = spline_net.spArgs.σ
    J = (h * λ[1]) .* Diagonal(σ'.(xInterm .+ b)) * W

    return eigvals(J)
end

## all layers spectrum check
function spectrum_check(spline_net::SplineNetwork)
    spArgs = spline_net.spArgs

    # state is a pointer to workspace sInterm
    state = spline_net.networkbackend.sInterm

    spectrums = rand(ComplexF64,spArgs.nLayers,spline_net.networkbackend.width)
    # loop to get all the state variables
    for i in 1:spArgs.nLayers
        state .= separate_slice(spline_net.networkbackend.states, i)
        t = (i - 1)*spArgs.hLayers                # "time" at (i-1)-th layer
        l = floor(Int, t/spArgs.δKnots)           # t ϵ [lδ, (l+1)δ]
        spectrums[i,:] .= layer_spectrum_check(spline_net, state, t, l)
    end
    return spectrums
end

#===============================
    For λ being trainable
================================#
## custom adjoint
function pullback_network(cx, spline_net::SplineNetwork, x)
    y = apply_network(spline_net::SplineNetwork, x)
    back = function(z)
        gradient_update!(spline_net, z)
        Zygote.accum_param(cx, spline_net.networkbackend.cfWeights, spline_net.networkbackend.gdWeights)
        Zygote.accum_param(cx, spline_net.networkbackend.cfBiases, spline_net.networkbackend.gdBiases)
        Zygote.accum_param(cx, spline_net.networkbackend.λ, spline_net.networkbackend.dλ)
        return ((cfWeights=spline_net.networkbackend.gdWeights,
                  cfBiases=spline_net.networkbackend.gdBiases,
                         λ=spline_net.networkbackend.dλ), spline_net.networkbackend.vInterm)
    end
    return y, back
end
## Tell Zygote about our custom adjoint
Zygote.@adjoint apply_network(spline_net::SplineNetwork, x) = pullback_network(__context__, spline_net, x)

## Constructors & trainable parameters
(spline_net::SplineNetwork)(x) = apply_network(spline_net, x)
Flux.trainable(spline_net::SplineNetwork) = (spline_net.networkbackend.cfWeights,
                                             spline_net.networkbackend.cfBiases,
                                             spline_net.networkbackend.λ,)

# #===============================
#     For λ being non-trainable
# ================================#
# ## custom adjoint
# function pullback_network(cx, spline_net::SplineNetwork, x)
#     y = apply_network(spline_net::SplineNetwork, x)
#     back = function(z)
#         gradient_update!(spline_net, z)
#         Zygote.accum_param(cx, spline_net.networkbackend.cfWeights, spline_net.networkbackend.gdWeights)
#         Zygote.accum_param(cx, spline_net.networkbackend.cfBiases, spline_net.networkbackend.gdBiases)
#         return ((cfWeights=spline_net.networkbackend.gdWeights,
#                   cfBiases=spline_net.networkbackend.gdBiases), spline_net.networkbackend.vInterm)
#     end
#     return y, back
# end
# ## Tell Zygote about our custom adjoint
# Zygote.@adjoint apply_network(spline_net::SplineNetwork, x) = pullback_network(__context__, spline_net, x)
#
# ## Constructors & trainable parameters
# (spline_net::SplineNetwork)(x) = apply_network(spline_net, x)
# Flux.trainable(spline_net::SplineNetwork) = (spline_net.networkbackend.cfWeights,
#                                              spline_net.networkbackend.cfBiases,)
