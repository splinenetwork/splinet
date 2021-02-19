
using DelimitedFiles, LinearAlgebra, Plots, Random
using Flux: mse
using Flux.Data: DataLoader

include("$(@__DIR__)/../src/spline_network.jl")
include("$(@__DIR__)/../src/train.jl")
include("$(@__DIR__)/../src/hyperopt.jl")


## Data loading function
function getdata(batchsize)

    ntrain = 100
    xtrain = readdlm("$(@__DIR__)/data/noisystep/x.dat")[1:ntrain,:]'
    ytrain = readdlm("$(@__DIR__)/data/noisystep/y.dat")[1:ntrain,:]'

    ntest = 100
    xtest = readdlm("$(@__DIR__)/data/noisystep/x.dat")[ntrain+1:ntrain+ntest,:]'
    ytest = readdlm("$(@__DIR__)/data/noisystep/y.dat")[ntrain+1:ntrain+ntest,:]'

    # Batching, data is shuffled already
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=false)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize, shuffle=false)

    return train_data, test_data
end


## Plotting
function plot_predictions(model)
    n = 1000
    x = range(-1.0, stop=1.0, length=n)
    y = map(xi -> model([xi])[1], x)
    plot(x, y, label="prediction")
    # plot!(x, sin.(x), label="exact")
end

# Opening layer copies x to network width
open_layer(w) = x -> repeat(x, w)
# Closing layer averages over network width
closing_layer(x) = mean(x, dims=1)


## Set up test case and call main training function
# This function is used by hyperopt.jl for tuning hyperparameters
function train_network(network_type::NetworkType{NT,D}; learning_rate=3e-3, epochs=100, batch_size=40, regularize=1e-9, network_width=2, nlayers=20, stepsize=0.4, nknots=10, amplitude=1.0, gnorm_tol=1e-4, do_plot=true, verbose=true) where {NT,D}

    # Specify activation function
    myactivation(x) = tanh(x)

    # Specify loss function
    myloss(x,y) = mse(x,y)

    # Target accuracy
    target_accuracy = 0.0

    # Set up Hidden layers
    nKnots = (NT == :spline) ? nknots : nlayers
    paras = SplineParameters{Float64,D}(nLayers=nlayers, hLayers=stepsize, width=network_width, nKnots=nknots, amplitude=amplitude, σ=myactivation)
    spline_layer = SplineNetwork(paras, init_weights_zeros=false, init_biases_zeros=false)

    # Set up network model: [Openinglayer, Hidden layers, closing layer]
    model = Chain(open_layer(network_width), spline_layer, closing_layer)

    # Train
    args = TrainingParameters(η=learning_rate, batchsize=batch_size, epochs=epochs, λ=regularize)
    final_accuracy_best = train(args; myloss=myloss, model=model, network_name=getName(network_type), target_accuracy=target_accuracy, gnorm_tol=gnorm_tol, do_plot=do_plot, verbose=verbose)

    return final_accuracy_best
end


# Create data set for noisy step function
# function createData(N, noiserange)
#     x = range(-1.0, stop=1.0, length=N)
#     y = zeros(N)
#     for i in 1:N
#           heavyside = (x[i] <= 0.0 ) ? 1.0 : -1.0
#           noise = -noiserange/2 + rand() * (noiserange)   # random number in [-0.2:0.2]
#           y[i] = heavyside + noise
#     end
#     return x, y
# end
#
