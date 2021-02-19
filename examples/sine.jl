using DelimitedFiles, LinearAlgebra, Plots
using Flux: mse
using Flux.Data: DataLoader


include("$(@__DIR__)/../src/unified_spline_network.jl")
include("$(@__DIR__)/../src/train.jl")
# include("$(@__DIR__)/../src/hyperopt.jl")

freq=5

## Data loading function
function getdata(batchsize)
    ntrain = freq*20*2
    case_str = "_k"*string(freq)
    xtrain = readdlm("$(@__DIR__)/data/sine/xtrain"*case_str*".dat")[1:ntrain,:]'
    ytrain = readdlm("$(@__DIR__)/data/sine/ytrain"*case_str*".dat")[1:ntrain,:]'
    # xtrain = readdlm("$(@__DIR__)/data/sine/x.dat")[1:ntrain,:]'
    # ytrain = readdlm("$(@__DIR__)/data/sine/y.dat")[1:ntrain,:]'

    ntest = 20
    xtest = readdlm("$(@__DIR__)/data/sine/xtest"*case_str*".dat")[1:ntest,:]'
    ytest = readdlm("$(@__DIR__)/data/sine/ytest"*case_str*".dat")[1:ntest,:]'
    # xtest = readdlm("$(@__DIR__)/data/sine/x.dat")[ntrain+1:ntrain+ntest,:]'
    # ytest = readdlm("$(@__DIR__)/data/sine/y.dat")[ntrain+1:ntrain+ntest,:]'

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=false)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize, shuffle=false)

    return train_data, test_data
end

## Overwrtie accuracy function
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        # acc += 1 - sum((model(x) - y).^2)/(2*sum(model(x).^2) + 2*sum(y.^2))
        acc += mse(model(x), y)
    end
    acc = acc / length(data_loader)
end

## Overwrtie prediction plotting function
function plot_predictions(model)
    n = 100
    x = range(-pi, stop=pi, length=n)
    y = map(xi -> model([xi])[1], x)
    plot(x, y, label="prediction", legend=:bottomright)
    plot!(x, sin.(freq*x), label="exact")
end

# Opening layer copies x to network width
open_layer(w) = x -> repeat(x, w)
# Closing layer averages over network width
closing_layer(x) = mean(x, dims=1)

## Set up test case and call main training function
# This function is used by hyperopt.jl for tuning hyperparameters
function train_network(network_type, deg=2; learning_rate=1e-2, epochs=1500,
                batch_size=5, regularize=1e-9, layer_dim=4, nlayers=100, stepsize=0.1,
                nknots=10, amplitude=1.0, init_λ=1.0, gnorm_tol=1e-5, do_plot=true, verbose=true)

    # Specify activation function
    myactivation(x) = tanh(x)

    # Specify loss function
    myloss(x,y) = mse(x,y)

    # Specify target accuracy, e.g. 1.0 (=100%) for peaks, 0.0 for sine-approximation, etc.
    target_accuracy = 0.0

    # Set up Hidden layers
    nKnots = (network_type == :spline) ? nknots : nlayers
    paras = SplineParameters{Float64,deg,typeof(myactivation)}(nLayers=nlayers, hLayers=stepsize,
                    nKnots=nknots, amplitude=amplitude, σ=myactivation)
    nnbackend = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ=init_λ,
                    init_weights_zeros=false, init_biases_zeros=false)
    spline_layer = SplineNetwork(paras, nnbackend)

    # Set up network model: [Openinglayer, Hidden layers, closing layer]
    model = Chain(open_layer(layer_dim), spline_layer, closing_layer)

    # Training paramters
    args = TrainingParameters(η=learning_rate, batchsize=batch_size, epochs=epochs, λ=regularize)
    # Call a general training function
    final_accuracy_best = train(args; myloss=myloss, model=model, network_name=network_type,
                    target_accuracy=target_accuracy, gnorm_tol=gnorm_tol, do_plot=do_plot, verbose=verbose)

    return final_accuracy_best
end

###################
# Example usages:
###################

## Train once
# train_network(:resnet, 1)
# train_network(:spline, 2)
# train_network(:spline, 2; epochs=1)


## Hyperparameter tuning
# resnet_hyperparam_tune(; nsamples=100, epochs=400)
# spline_hyperparam_tune(; degree=2, nsamples=2, epochs=1)

## Create boxplots:
# stats_plot("plots/resnet1_hyperopt_stats.jld", "Resnet")
# stats_plot("plots/spline2_hyperopt_stats.jld", "Spline2")


###################################
# Delta-t test for Spline network
###################################

 function dttest(degree)

    myactivation(x) = tanh(x)
    myloss(x,y) = mse(x,y)
     layer_dim=2
     batch_size=2
     init_λ = 1.0
     amp = 1.0

     # Generate finest problem
     T = 2.0
     h = 1e-7
     nknots = 20
     nlayers= round(Int64, T/h)  # => 10^6 layers!
     println("Ground truth with h= ", h, " nlayers=", nlayers)

    # Set up Hidden layers
    paras = SplineParameters{Float64,degree,typeof(myactivation)}(nLayers=nlayers, hLayers=h,
                    nKnots=nknots, amplitude=amp, σ=myactivation)
    nnbackend_finest = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ=init_λ,
                    init_weights_zeros=false, init_biases_zeros=false)
    spline_layer = SplineNetwork(paras, nnbackend_finest)
    # Set up network model: [Openinglayer, Hidden layers, closing layer]
    model = Chain(open_layer(layer_dim), spline_layer, closing_layer)
    # Training paramters
    # args = TrainingParameters(η=0.0177142, batchsize=20, epochs=600, λ=8.43674e-10)

     # Store the weights and biases
     weights_store = nnbackend_finest.cfWeights
     biases_store  = nnbackend_finest.cfBiases

     # generate data x and ground truth
     x = range(-1.0, stop=1.0, length=batch_size)
     # y_truth = model(x)
     y_truth = map(xi -> model([xi])[1], x)
     println("y_truth: ", y_truth)

      # Iterate for coarser h
      err_vec = zeros(0)
      h_vec = zeros(0)
      for i in 1:10

         # Increase h
         h = h * 10.0
         nlayers = round(Int64, T/h)
         # No less than 10 layers
         if nlayers < 1
             break
         end

         # Generate next coarser model
         paras = SplineParameters{Float64,degree,typeof(myactivation)}(nLayers=nlayers, hLayers=h,
                    nKnots=nknots, amplitude=amp, σ=myactivation)
         nnbackend = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ=init_λ,
                    init_weights_zeros=false, init_biases_zeros=false)
         # Set the weights from the finest model
         nnbackend.cfWeights = weights_store
         nnbackend.cfBiases  = biases_store
         spline_layer = SplineNetwork(paras, nnbackend)
         model = Chain(open_layer(layer_dim), spline_layer, closing_layer)

         # Propagate forward and compute error
         # y = model(x)

         y = map(xi -> model([xi])[1], x)
         abs_err = norm((y - y_truth), 2)
         println(h, "   ", abs_err)
         append!(err_vec, abs_err)
         append!(h_vec, h)

     end

     println("stepsize   = ", h_vec)
     println("abs. error = ", err_vec)
     return h_vec, err_vec
 end

#  #Do the delta t test for degree 2 and 3
#   h_vec_S2, err_vec_S2 = dttest(2);
#   h_vec_S3, err_vec_S3 = dttest(3);
#   plot(h_vec_S2, err_vec_S2, yaxis=:log, xaxis=:log, marker=:d, legend=:bottomright, label="SpliNet, d=2", xlabel="h", ylabel="error of network output")
#   plot!(h_vec_S3, err_vec_S3, yaxis=:log, xaxis=:log, marker=:d, legend=:bottomright, label="SpliNet, d=3", xlabel="h", ylabel="error of network output")
#  savefig("plots/sine_dttest_T2_nknots20_randominit.png")
