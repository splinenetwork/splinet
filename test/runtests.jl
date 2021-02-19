using Test, Random, Printf

include("gradient_check.jl")

@testset "Spline Network Gradients" begin
    Random.seed!(0)

    e1, e2 = test_by_simple_network()
    @test e1 ≈ 0.0 atol=100*eps(Float64)
    @test e2 ≈ 0.0 atol=100*eps(Float64)

    atol = 1e-6 # Tolerance for finite difference comparisosn

    @test test_loss_by_central_difference() ≈ 0.0 atol=atol

    e1, e2 = test_by_central_difference()
    @test e1 ≈ 0.0 atol=atol
    @test e2 ≈ 0.0 atol=atol
end

@testset "Spline Network Performance and Allocations" begin
    bm = performance_test_forward()
    @test allocs(bm) == 0
    @printf("Forward pass (median,min,max): (%s, %s, %s)\n",
        BenchmarkTools.prettytime(time(median(bm))),
        BenchmarkTools.prettytime(time(minimum(bm))),
        BenchmarkTools.prettytime(time(maximum(bm))))
    bm = performance_test_backward()
    @test allocs(bm) == 0
    @printf("Backward pass (median,min,max): (%s, %s, %s)\n",
        BenchmarkTools.prettytime(time(median(bm))),
        BenchmarkTools.prettytime(time(minimum(bm))),
        BenchmarkTools.prettytime(time(maximum(bm))))
end

@testset "Spline Network Resnet Comparison" begin
    spline_net,x,_ = setup_test_network(deg=1, batch_size=10, width=10, nLayers=10, nKnots=10)

    nlayers = spline_net.args.nLayers
    h = spline_net.args.hLayers
    w = spline_net.args.width

    layers = []
    for i=1:nlayers
        W = spline_net.cfWeights[:,:,i]
        b = spline_net.cfBiases[:,i]
        layer = SkipConnection(Dense(W, b, relu), (mx, x) -> x + h*mx)
        push!(layers, layer)
    end
    resnet = Chain(layers...)

    @test norm(spline_net(x)-resnet(x),Inf) == 0.0
end

stability_check(Float64; batch_size = 10)
