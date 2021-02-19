using Test

include("../src/unified_spline_network.jl")


function spline_conv_net_test(::Type{T} = Float64) where T
    nlayers = 10
    hlayers = T(0.05)
    nBatch = 10
    dataDims = (20,20,1,1,nBatch)
    conv_net = []
    filterDims = (3,3,1)

    τ = 1.0
    for i=1:nlayers
        w = glorot_uniform_tensor(T, filterDims..., 1, 1)
        b = zeros(T, 1)
        conv_layer = Conv(w, b, relu; pad=floor.(Int, filterDims./2), stride=1)
        layer = SkipConnection(conv_layer,(cx, x) -> x .+ (τ * hlayers) .*cx)
        push!(conv_net, layer)
    end
    model_flux = Chain(conv_net...)

    xdata = rand(T,dataDims...)
    label = rand(T,dataDims...)

    pf = Flux.params(model_flux)
    loss(x) = sum((model_flux(x) - label).^2)
    gf = gradient(() -> loss(xdata), pf)

    convArgs = ConvParameters(filterDims=filterDims, stride=1)
    args = SplineParameters{T,1}(nKnots=nlayers, nLayers=nlayers, hLayers=hlayers)
    backend = ConvNetworkBackend(args, convArgs; dataDims=dataDims[1:4], batchSize = nBatch,
                        init_biases_zeros=true)
    model_my = SplineNetwork(args, backend)

    for i in 1:nlayers
        separate_slice(model_my.networkbackend.cfWeights, i) .= Flux.params(model_flux)[2*i-1]
    end

    ps = Flux.params(model_my)
    loss2(x) = sum((model_my(x) - label).^2)
    gs = gradient(() -> loss2(xdata), ps)

    err_for = sum(abs.( model_flux(xdata) - model_my(xdata)))
    err_grad = 0.0
    for i in 1:nlayers
        err_grad += norm(gs[ps[1]][:,:,:,:,:,i] - gf[pf[2*i-1]], Inf)
    end
    return err_for/nBatch, err_grad/nBatch
end

function spline_dense_net_test(::Type{T} = Float64) where T
    nlayers = 10
    hlayers = T(0.05)
    nBatch = 10
    dense_net = []
    width = 10

    for i=1:nlayers
        w = glorot_uniform_tensor(T, width, width)
        b = zeros(T, width)
        denselayer = Dense(w,b,relu)
        layer = SkipConnection(denselayer, (σx, x) -> x .+ hlayers.*σx)
        push!(dense_net, layer)
    end
    model_flux = Chain(dense_net...)

    xdata = rand(T,width,nBatch)
    label = rand(T,width,nBatch)

    pf = Flux.params(model_flux)
    loss(x) = sum((model_flux(x) - label).^2)
    gf = gradient(() -> loss(xdata), pf)

    args = SplineParameters{T,1}(nKnots = nlayers, nLayers = nlayers, hLayers = hlayers)
    backend = DenseNetworkBackend(width, args; batchSize = nBatch)
    model_my = SplineNetwork(args, backend)

    for i in 1:nlayers
        separate_slice(model_my.networkbackend.cfWeights, i) .= Flux.params(model_flux)[2*i-1]
        separate_slice(model_my.networkbackend.cfBiases, i) .= Flux.params(model_flux)[2*i]
    end

    ps = Flux.params(model_my)
    loss2(x) = sum((model_my(x) - label).^2)
    gs = gradient(() -> loss2(xdata), ps)

    err_for = sum(abs.( model_flux(xdata) - model_my(xdata)))
    err_grad = 0.0
    for i in 1:nlayers
        err_grad += norm(gs[ps[1]][:,:,i] - gf[pf[2*i-1]], Inf)
        err_grad += norm(gs[ps[2]][:,i] - gf[pf[2*i]], Inf)
    end
    return err_for/nBatch, err_grad/nBatch
end


function spline_conv_net_τtest(::Type{T} = Float64) where T
    nlayers = 10
    hlayers = T(0.05)
    nBatch = 10
    dataDims = (20,20,1,1,nBatch)
    filterDims = (3,3,1)

    conv_net_le = []
    conv_net_ri = []
    τ = 1.0
    Δ = T(1e-5)
    for i=1:nlayers
        w = glorot_uniform_tensor(T, filterDims..., 1, 1)
        b = zeros(T, 1)
        conv_layer = Conv(w, b, relu; pad=floor.(Int, filterDims./2), stride=1)
        layer_le = SkipConnection(conv_layer,(cx, x) -> x .+ ((τ-Δ) * hlayers) .*cx)
        layer_ri = SkipConnection(conv_layer,(cx, x) -> x .+ ((τ+Δ) * hlayers) .*cx)
        push!(conv_net_le, layer_le)
        push!(conv_net_ri, layer_ri)
    end
    model_flux_le = Chain(conv_net_le...)
    model_flux_ri = Chain(conv_net_ri...)

    xdata = rand(T,dataDims...)
    label = rand(T,dataDims...)


    loss_le = sum((model_flux_le(xdata) - label).^2)
    loss_ri = sum((model_flux_ri(xdata) - label).^2)
    gf = (loss_ri - loss_le)/(2*Δ)


    convArgs = ConvParameters(filterDims=filterDims, stride=1)
    args = SplineParameters{T,1}(nKnots=nlayers, nLayers=nlayers, hLayers=hlayers)
    backend = ConvNetworkBackend(args, convArgs; dataDims=dataDims[1:4], batchSize = nBatch,
                        init_biases_zeros=true)
    model_my = SplineNetwork(args, backend)

    for i in 1:nlayers
        separate_slice(model_my.networkbackend.cfWeights, i) .= Flux.params(model_flux_le)[2*i-1]
    end

    ps = Flux.params(model_my)
    loss2(x) = sum((model_my(x) - label).^2)
    gs = gradient(() -> loss2(xdata), ps)

    err_τ = abs( gs[ps[3]][1] - gf )
    return err_τ
end

function spline_dense_net_τtest(::Type{T} = Float64) where T
    nlayers = 10
    hlayers = T(0.05)
    nBatch = 10
    width = 10

    dense_net_le = []
    dense_net_ri = []
    τ = 1.0
    Δ = T(1e-5)
    for i=1:nlayers
        w = glorot_uniform_tensor(T, width, width)
        b = zeros(T, width)
        denselayer = Dense(w,b,relu)
        layer_le = SkipConnection(denselayer, (σx, x) -> x .+ ((τ-Δ)*hlayers).*σx)
        layer_ri = SkipConnection(denselayer, (σx, x) -> x .+ ((τ+Δ)*hlayers).*σx)
        push!(dense_net_le, layer_le)
        push!(dense_net_ri, layer_ri)
    end
    model_flux_le = Chain(dense_net_le...)
    model_flux_ri = Chain(dense_net_ri...)

    xdata = rand(T,width,nBatch)
    label = rand(T,width,nBatch)

    loss_le = sum((model_flux_le(xdata) - label).^2)
    loss_ri = sum((model_flux_ri(xdata) - label).^2)
    gf = (loss_ri - loss_le)/(2*Δ)

    args = SplineParameters{T,1}(nKnots = nlayers, nLayers = nlayers, hLayers = hlayers)
    backend = DenseNetworkBackend(width, args; batchSize = nBatch)
    model_my = SplineNetwork(args, backend)

    for i in 1:nlayers
        separate_slice(model_my.networkbackend.cfWeights, i) .= Flux.params(model_flux_le)[2*i-1]
        separate_slice(model_my.networkbackend.cfBiases, i) .= Flux.params(model_flux_le)[2*i]
    end

    ps = Flux.params(model_my)
    loss2(x) = sum((model_my(x) - label).^2)
    gs = gradient(() -> loss2(xdata), ps)

    err_τ = abs( gs[ps[3]][1] - gf )
    return err_τ
end


@testset "Spline Network" begin
    atol = sqrt(eps())
    err_for, err_grad = spline_conv_net_test()
    @test err_for ≈ 0.0 atol=atol
    @test err_grad ≈ 0.0 atol=atol

    err_for, err_grad = spline_dense_net_test()
    @test err_for ≈ 0.0 atol=atol
    @test err_grad ≈ 0.0 atol=atol

    err_τ = spline_conv_net_τtest()
    @test err_τ ≈ 0.0 atol=atol
    err_τ = spline_dense_net_τtest()
    @test err_τ ≈ 0.0 atol=atol
end
