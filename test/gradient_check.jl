using Flux, Zygote, LinearAlgebra, Statistics
using Flux: logitcrossentropy

include("spline_network.jl")
# include("quadratic_splines.jl")
## a general function of central difference
#=      f:  a callable struct
        x:  input
    paras:  parameters whose gradients need to be computed
    o_dim:  length of f's output
=#
function gradients_central_difference(f, paras::Params, o_dim) where T
    gradients = ()
    scalar_central_difference(yp, yn, h) = (yp - yn)/(2*h)
    for p in paras
        gdAllEntry  = ()
        for k in 1:o_dim
            gd = similar(p)
            for i in eachindex(p)
                z = p[i]
                h = convert(typeof(z), 1e-4)
                p[i] = z + h
                ypos = f()[k]
                p[i] = z - h
                yneg = f()[k]

                gd[i] = scalar_central_difference(ypos, yneg, h)
                p[i] = z
            end
            gdAllEntry = (gdAllEntry..., gd)
        end
        gradients = (gradients..., gdAllEntry)
    end
    return gradients
end

function compare_gradient_central_differences(f, params, T::DataType = Float64)
    g1 = gradient(f, params)
    g2 = gradients_central_difference(f, params, 1)

    error = 0.0
    idx = 1
    for p in params
        error = max(error, norm(g1[p]-g2[idx][1], Inf))
        idx += 1
    end
    return error
end

## test by central difference
function test_by_central_difference(T::DataType = Float64; batch_size=4)
    paras = SplineParameters{T,3}(nLayers=30, hLayers=0.05, width=10, nKnots=10)
    spline_net = SplineNetwork(paras)

    n = paras.width
    x = rand(T, n, batch_size)
    p = Flux.params(spline_net.cfWeights, spline_net.cfBiases)
    dspline_net = gradients_central_difference(() -> spline_net(x), p, n*batch_size)

    diffW, diffB = 0.0, 0.0
    z = zeros(T, n, batch_size)
    for b in 1:batch_size
        for k in 1:n
            spline_net(x)
            z[k,b] = 1.0
            gradient_update!(spline_net, z)
            z[k,b] = 0.0

            for i in 1:paras.nKnots
                diffW = max(diffW, norm(dspline_net[1][k+(b-1)*n][:,:,i] - spline_net.gdWeights[:,:,i], Inf) / (eps(T) + norm(dspline_net[1][k+(b-1)*n][:,:,i], Inf)))
                diffB = max(diffW, norm(dspline_net[2][k+(b-1)*n][:,i] - spline_net.gdBiases[:,i], Inf) / (eps(T) + norm(dspline_net[2][k+(b-1)*n][:,i], Inf)))
            end
        end
    end

    return diffW, diffB
end

## test by central difference
function test_loss_by_central_difference(T::DataType = Float64; batch_size=4)
    sz_in = 5
    sz_out = 7
    width = 10
    paras = SplineParameters{T,3}(nLayers=100, hLayers=0.01, width=width, nKnots=8)
    spline_net = SplineNetwork(paras)

    mkdense(w,h) = Dense(rand(T,h,w), rand(T,h))

    model = Chain(mkdense(sz_in,width),spline_net,mkdense(width,sz_out))

    x = rand(T, sz_in, batch_size)
    y = rand(T, sz_out, batch_size)

    p = Flux.params(model)
    loss() = logitcrossentropy(model(x), y) + mean(norm, p)
    compare_gradient_central_differences(loss, p)
end

## Simple, explicitly written-out network
struct SimpleNetwork{T}
    args::SplineParameters
    w1::Matrix{T}
    w2::Matrix{T}
    w3::Matrix{T}
    w4::Matrix{T}
    b1::Vector{T}
    b2::Vector{T}
    b3::Vector{T}
    b4::Vector{T}
end

function (nn::SimpleNetwork{T})(x) where T
    δ = nn.args.δKnots
    h = nn.args.hLayers
    u1 = x .+ h * relu.( (Bk(T(0),1,δ)*nn.w1 + Bk(T(0),2,δ)*nn.w2 + Bk(T(0),3,δ)*nn.w3) * x .+
                      Bk(T(0),1,δ)*nn.b1 + Bk(T(0),2,δ)*nn.b2 + Bk(T(0),3,δ)*nn.b3 )
    u2 = u1 .+ h * relu.( (Bk(h,1,δ)*nn.w1 + Bk(h,2,δ)*nn.w2 + Bk(h,3,δ)*nn.w3) * u1 .+
                      Bk(h,1,δ)*nn.b1 + Bk(h,2,δ)*nn.b2 + Bk(h,3,δ)*nn.b3 )
    u3 = u2 .+ h * relu.( (Bk(2*h,2,δ)*nn.w2 + Bk(2*h,3,δ)*nn.w3 + Bk(2*h,4,δ)*nn.w4) * u2 .+
                       Bk(2*h,2,δ)*nn.b2 + Bk(2*h,3,δ)*nn.b3 + Bk(2*h,4,δ)*nn.b4 )
    return u3
end

## gradient correctness check by a simple network
function test_by_simple_network(T::DataType = Float64)
    # construct an instance
    paras = SplineParameters{T}(nLayers = 3, hLayers=1/3, width = 2, nKnots = 2)
    spline_net = SplineNetwork(paras)
    n = paras.width
    # forward pass
    x = randn(T, n)
    spline_net(x)
    # Define some random data and loss function
    y = randn(T, n)
    loss(spline_net, x, y) = sum((spline_net(x) - y).^2)
    # Compute the gradient of the network using our custom gradients
    gtest = gradient(m -> loss(m, x, y), spline_net)[1]

    # test with a simple network
    simple_net = SimpleNetwork(paras, spline_net.cfWeights[:,:,1], spline_net.cfWeights[:,:,2], spline_net.cfWeights[:,:,3], spline_net.cfWeights[:,:,4],
                       spline_net.cfBiases[:,1], spline_net.cfBiases[:,2], spline_net.cfBiases[:,3], spline_net.cfBiases[:,4])
    p_sn = Flux.params(simple_net.w1, simple_net.w2, simple_net.w3, simple_net.w4, simple_net.b1, simple_net.b2, simple_net.b3, simple_net.b4,)
    gtrue = gradient(() -> loss(simple_net, x, y), p_sn)

    diffW, diffB = 0.0, 0.0
    gtrueW = T(0.0) .* gtest[1]
    gtrueB = T(0.0) .* gtest[2]
    for i in 1:4
        diffW += norm(gtest[1][:,:,i] - gtrue[p_sn[i]], Inf) / ( eps(T) + norm(gtrue[p_sn[i]], Inf) )
        gtrueW[:,:,i] = gtrue[p_sn[i]]
    end
    for i in 5:8
        diffB += norm(gtest[2][:,i-4] - gtrue[p_sn[i]], Inf) / ( eps(T) + norm(gtrue[p_sn[i]], Inf) )
        gtrueB[:,i-4] = gtrue[p_sn[i]]
    end
    # gtest, (gtrueW, gtrueB),
    return  diffW, diffB
end

function setup_test_network(T::DataType = Float32; deg=2, batch_size=40, kwargs...)
    spline_net = SplineNetwork(SplineParameters{T,deg}(;kwargs...))
    n = spline_net.args.width
    x = randn(T, n, batch_size)
    y = randn(T, n, batch_size)
    return spline_net, x, y
end

using Plots
## check the stability of Forward-Euler
function stability_check_different_h(T::DataType = Float32, coarsest = 100, ntest = 5; batch_size = 40, kwargs...)
    nwidth = 10
    knots = 10
    x = randn(T, nwidth, batch_size)

    spline_layer_finest = SplineNetwork(SplineParameters{T}(nLayers=10000, hLayers=1.0/10000, width=nwidth, nKnots=knots), init_weights_zeros=false, init_biases_zeros=false)
    output_finest = spline_layer_finest(x)

    err_vec = zeros(ntest,)
    for i in 1:5
        spline_layer_test = SplineNetwork(SplineParameters{T}(nLayers=coarsest*(2^i), hLayers=1.0/(coarsest*(2^i)), width=nwidth, nKnots=knots), init_weights_zeros=false, init_biases_zeros=false)
        spline_layer_test.cfWeights = spline_layer_finest.cfWeights
        spline_layer_test.cfBiases  = spline_layer_finest.cfBiases

        err_vec[i] = norm(output_finest - spline_layer_test(x), Inf)
    end
    return err_vec
end

function stability_check(T::DataType = Float32, ; batch_size = 40, kwargs...)
    coarsest = 100
    ntest = 5
    h_vec = log10.( 1.0 ./ (coarsest .* 2 .^ (1:ntest)) )
    err_vec = log10.( stability_check_different_h(Float64, coarsest, ntest; batch_size = 10) )
    one_slope = (err_vec[1] - h_vec[1]) .+ h_vec
    plot(
        h_vec,
        err_vec,
        size=(500,400),
        marker=:d,
        label="spline_network",
        legend=:bottomright,
        xlabel="log_10(h)",
        ylabel="log_10(|Δu|)",
    )
    display(plot!(h_vec, one_slope, label="slope one"))
    savefig("$(@__DIR__)/../examples/plots/stability_forward_euler.png")
end

function performance_test_forward(T::DataType = Float32; kwargs...)
    spline_net, x, z = setup_test_network(T; kwargs...)
    @benchmark $spline_net($x)
end

function performance_test_backward(T::DataType = Float32; kwargs...)
    spline_net, x, z = setup_test_network(T; kwargs...)
    spline_net(x)
    @benchmark gradient_update!($spline_net, $z)
end
