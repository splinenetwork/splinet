using Flux, Zygote, BenchmarkTools, LinearAlgebra, UnsafeArrays
using Base: @kwdef

## parameters
Base.@kwdef mutable struct Parameters{T,F}
    width::Int = 220
    σ::F = relu
    h::T = 0.05
    filterDim::Int = 3
end
mutable struct SimpleConvNetwork{T,F}
    args::Parameters{T,F}
    # filter & its gradient
    K::Array{T,3}
    dK::Array{T,3}
    batchSize::Int
    # workspace used for computations
    xInterm::Array{T,3}
    vInterm::Array{T,3}
    zInterm::Array{T,3}
    zConVx::Array{T,3}
end

function SimpleConvNetwork(args::Parameters{T,F}; batchSize=1) where {T,F}
    filterDim = args.filterDim
    width = args.width
    # initialization
    return SimpleConvNetwork{T,F}(
        args,
        # filter & its gradient
        zeros(T, filterDim, 1, 1),
        zeros(T, filterDim, 1, 1),
        batchSize,
        # workspace used for computations
        zeros(T, width, 1, batchSize),
        zeros(T, width, 1, batchSize),
        zeros(T, width, 1, batchSize),
        zeros(T, filterDim, 1, 1),
    )
end

function apply_network(conv_net::SimpleConvNetwork, x::AbstractArray)
    xInterm = conv_net.xInterm
    K = conv_net.K
    cdims = DenseConvDims(size(x), size(K); stride=1, padding=1)
    conv!(xInterm, x, K, cdims)
    h = conv_net.args.h
    σ = conv_net.args.σ
    return x .+ h .* σ.(xInterm)
end

function gradient_update!(conv_net::SimpleConvNetwork, x::AbstractArray, v::AbstractArray)
    # In this function we need to compute
    #    ∇_x( x + h σ(K * x) ) as the adjoint
    #    ∇_K( x + h σ(K * x) ) as the gradient to K
    # Both x & v are 3d arrays, arranged as dim x channel x batchsize
    xInterm = conv_net.xInterm
    K = conv_net.K
    cdims = DenseConvDims(size(x), size(K); stride=1, padding=1)
    # This will give us K * x
    conv!(xInterm, x, K, cdims)


    zInterm = conv_net.zInterm
    σ = conv_net.args.σ
    zInterm .= σ'.(xInterm) .* v

    h = conv_net.args.h
    zConVx = conv_net.zConVx

    for b in 1:conv_net.batchSize
        z = reshape(zInterm[:,1,b], size(zInterm[:,1,b],1), 1, 1)
        u = reshape(x[:,1,b], size(x[:,1,b],1), 1, 1)
        ∇conv_filter!(zConVx, u, z, cdims)
        conv_net.dK .+= h .* zConVx
    end

    cdims = DenseConvDims(size(zInterm), size(K); stride=1, padding=1)
    ∇conv_data!(xInterm, zInterm, K, cdims)

    conv_net.vInterm .= v .+  h .* xInterm
end

# custom adjoint
function pullback_network(cx, conv_net::SimpleConvNetwork, x)
    y = apply_network(conv_net::SimpleConvNetwork, x)
    back = function(z)
        gradient_update!(conv_net, x, z)
        Zygote.accum_param(cx, conv_net.K, conv_net.dK)
        return ((K=conv_net.dK,), conv_net.vInterm)
    end
    return y, back
end
# Tell Zygote about our custom adjoint
Zygote.@adjoint apply_network(conv_net::SimpleConvNetwork, x) = pullback_network(__context__, conv_net, x)

## Constructors & trainable parameters
(conv_net::SimpleConvNetwork)(x) = apply_network(conv_net, x)
Flux.trainable(conv_net::SimpleConvNetwork) = (conv_net.K,)

## main test
function simple_conv_net_test()
    h = Float32(0.05)
    conv_layer = Conv((3,), 1=>1, relu; pad=1, stride=1)
    flux_model = SkipConnection(conv_layer,(cx, x) -> x .+ h.*cx)

    nBatch = 10
    label = rand(Float32,220,1,nBatch)
    loss(x) = sum((flux_model(x) - label).^2)

    xdata = rand(Float32,220,1,nBatch)
    pf = Flux.params(flux_model)
    gf = gradient(() -> loss(xdata), pf)

    args = Parameters{Float32,typeof(relu)}(h = Float32(0.05), filterDim=3)
    my_model = SimpleConvNetwork(args; batchSize=nBatch)
    my_model.K = pf[1]

    ps = Flux.params(my_model)
    loss2(x) = sum((my_model(x) - label).^2)
    gs = gradient(() -> loss2(xdata), ps)


    err_forw = sum(abs.(flux_model(xdata) - my_model(xdata)))
    err_grad = norm(gs[ps[1]] - gf[pf[1]], Inf)
    return err_forw/nBatch, err_grad/nBatch
end

function test_hundred_times()
    err_forw = 0.0
    err_grad = 0.0
    for i in 1:100
        e1, e2 = simple_conv_net_test()
        err_forw += e1
        err_grad += e2
    end
    return err_forw/100, err_grad/100
end

err_forw, err_grad = test_hundred_times()
