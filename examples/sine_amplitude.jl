using DelimitedFiles, LinearAlgebra, Plots, Random
using Flux: mse
using Flux.Data: DataLoader

include("$(@__DIR__)/../src/unified_spline_network.jl")
include("$(@__DIR__)/../src/train.jl")

f(x) = 10.0*sin(3.0*x)

## Data loading function
function getdata(batchsize)
    ntrain = 750
    xtrain = 2*pi*(rand(ntrain)'.-0.5)
    ytrain = f.(xtrain)

    ntest = 750
    xtest = 2*pi*(rand(ntest)'.-0.5)
    ytest = f.(xtest)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=false)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize, shuffle=false)

    return train_data, test_data
end

## Overwrite accuracy function
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += 1 - sum((model(x) - y).^2)/(2*sum(model(x).^2) + 2*sum(y.^2))
    end
    acc = acc / length(data_loader)
end

## Overwrtie prediction plotting function
function plot_predictions(model)
    n = 1000
    x = range(-pi, stop=pi, length=n)
    y = map(xi -> model([xi])[1], x)
    plot(x, y, label="prediction", legend=:bottomright)
    y_ex = f.(x)
    plot!(x, y_ex, label="exact", ylim=extrema(y_ex))
end

# Opening layer copies x to network width
open_layer(w) = x -> repeat(x, w)
# Closing layer averages over network width
closing_layer(x) = mean(x, dims=1)

## Set up test case and call main training function
# This function is used by hyperopt.jl for tuning hyperparameters
function train_network(network_type, deg=1; learning_rate=1e-2, epochs=200, λ=1.0,
                batch_size=5, regularize=1e-9, layer_dim=2, nlayers=20,
                nknots=10, amplitude=1.0, gnorm_tol=1e-4, do_plot=true, verbose=true)

    σ(x) = tanh(x) # activation function
    myloss(x,y) = mse(x,y) # loss function

    target_accuracy = 1.0

    # Set up Hidden layers
    nKnots = (network_type == :spline) ? nknots : nlayers
    h = 1.0/nlayers
    paras = SplineParameters{Float64,deg,typeof(σ)}(nLayers=nlayers, hLayers=h,
                    nKnots=nknots, amplitude=amplitude, σ=σ)
    # nnbackend = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size,
    #                 init_weights_zeros=false, init_biases_zeros=false, init_λ=λ)
    nnbackend = DenseAntisymmetricNetworkBackend(layer_dim, paras, batchSize=batch_size,
                init_weights_zeros=false, init_biases_zeros=false, init_λ=λ)
    spline_layer = SplineNetwork(paras, nnbackend)

    # Set up network model: [Openinglayer, Hidden layers, closing layer]
    model = Chain(open_layer(layer_dim), spline_layer, closing_layer)

    # Training paramters
    args = TrainingParameters(η=learning_rate, batchsize=batch_size, epochs=epochs, λ=regularize)
    # Call a general training function
    final_accuracy_best = train(args; myloss=myloss, model=model, network_name=network_type,
                    target_accuracy=target_accuracy, gnorm_tol=gnorm_tol, do_plot=do_plot, verbose=verbose)

    model, final_accuracy_best
end

function plot_bounds(m)
    plot_predictions(m)
    plot!(range(-pi,pi,length=2), x -> x-3, color=:black, label=raw"$x \pm 3$")
    plot!(range(-pi,pi,length=2), x -> x+3, label=nothing, color=:black)
end

function run_samples(n)
    accuracies = []
    λ = []
    for i=1:n
        m,acc = train_network(:spline, 2, nknots=10, nlayers=20, batch_size=50, epochs=200, layer_dim=4, λ=3.0, do_plot=false);
        push!(accuracies, acc)
        push!(λ, m[2].networkbackend.λ)
    end
    accuracies,λ
end
