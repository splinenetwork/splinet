using DelimitedFiles, LinearAlgebra, Plots, Random
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Data: DataLoader


include("$(@__DIR__)/../src/unified_spline_network.jl")
include("$(@__DIR__)/../src/train.jl")
# include("$(@__DIR__)/../src/hyperopt.jl")

# Specify dimension of input and output data
const data_dim_in = 2
const data_dim_out = 5

## Data loading function
function getdata(batchsize)
    # Loading Dataset
    xdata = readdlm("$(@__DIR__)/data/peaks/features.dat")
    ydata = readdlm("$(@__DIR__)/data/peaks/labels.dat")

    # Randomly shuffle the data and labels
    perm = randperm(size(xdata,1))
    xdata = xdata[perm,:]'
    ydata = ydata[perm,:]'

    ntrain = 1000

    xtrain = xdata[:,1:ntrain]
    ytrain = ydata[:,1:ntrain]

    ntest = 1000
    xtest = xdata[:,(ntrain+1):ntrain+ntest]
    ytest = ydata[:,(ntrain+1):ntrain+ntest]

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize)

    return train_data, test_data
end

## Overwrtie prediction plotting function
function plot_predictions(model)
    n = 128
    x = range(-3.0, stop=3.0, length=n) |> collect
    xx = repeat(x, length(x))
    yy = repeat(x, inner=length(x))
    xeval = transpose(hcat(xx, yy))
    pred = onecold(model(xeval))
    heatmap(x, x, reshape(pred,n,n)',
        # seriescolor=cgrad(:jet, data_dim_out, rev=true, categorical=true),
        seriescolor=cgrad(:magma, data_dim_out),
        aspect_ratio=:equal, titlefontsize=20, legend=:none)
end

## Overwrtie accuracy function
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y)) * 1 / size(x,2)
    end
    acc/length(data_loader)
end

## Set up test case and call main training function
# This function is used by hyperopt.jl for tuning hyperparameters
function train_network(network_type, deg=1; learning_rate=3e-3, epochs=100,
                batch_size=40, regularize=1e-9, layer_dim=5, nlayers=20, stepsize=0.4,
                nknots=10, amplitude=1.0, gnorm_tol=1e-4, init_λ=1.0, do_plot=true, verbose=true)

    # Specify activation function
    myactivation(x) = relu(x)

    # Specify loss function
    myloss(x,y) = logitcrossentropy(x,y)

    # Specify target accuracy, 1.0 (=100%) for peaks
    target_accuracy = 1.0

    # Set up Hidden layers
    nKnots = (network_type == :spline) ? nknots : nlayers
    paras = SplineParameters{Float64,deg,typeof(myactivation)}(nLayers=nlayers, hLayers=stepsize,
                    nKnots=nknots, amplitude=amplitude, σ=myactivation)
    nnbackend = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ = init_λ,
                    init_weights_zeros=false, init_biases_zeros=false)
    spline_layer = SplineNetwork(paras, nnbackend)

    # Set up opening and cosing weights and biases
    # Random.seed!(0)
    W_in  = Float32(amplitude) .* rand(Float32, layer_dim, data_dim_in)
    b_in  = Float32(amplitude) .* rand(Float32, layer_dim)
    W_out = Float32(amplitude) .* rand(Float32, data_dim_out, layer_dim)
    b_out = Float32(amplitude) .* rand(Float32, data_dim_out)

    # Set up network model: [Opening layer, Hidden layers, closing layer]
    model = Chain(Dense(W_in, b_in), spline_layer, Dense(W_out, b_out))

    # Training parameters
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
# resnet_hyperparam_tune(; nsamples=2, epochs=1)
# spline_hyperparam_tune(; degree=2, nsamples=2, epochs=1)

## Create boxplots:
# stats_plot("plots/resnet1_hyperopt_stats.jld", "Resnet")
# stats_plot("plots/spline2_hyperopt_stats.jld", "Spline2")


###################################
# Delta-t test for Spline network
###################################

# function dttest(degree)
#
#    myactivation(x) = relu(x)
#    myloss(x,y) = logitcrossentropy(x,y)
#     layer_dim=5
#     batch_size=1
#     init_λ = 1.0
#     amp = 0.1
#
#     # Generate finest problem
#     T = 2.0
#     h = 1e-7
#     nknots = 20
#     nlayers= round(Int64, T/h)  # => 10^6 layers!
#     println("Ground truth with h= ", h, " nlayers=", nlayers)
#
#    # Set up Hidden layers
#    paras = SplineParameters{Float64,degree,typeof(myactivation)}(nLayers=nlayers, hLayers=h,
#                    nKnots=nknots, amplitude=amp, σ=myactivation)
#    nnbackend_finest = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ=init_λ,
#                    init_weights_zeros=false, init_biases_zeros=false)
#    spline_layer = SplineNetwork(paras, nnbackend_finest)
#    W_in  = Float32(amp) .* rand(Float32, layer_dim, data_dim_in)
#    b_in  = Float32(amp) .* rand(Float32, layer_dim)
#    W_out = Float32(amp) .* rand(Float32, data_dim_out, layer_dim)
#    b_out = Float32(amp) .* rand(Float32, data_dim_out)
#    model = Chain(Dense(W_in, b_in), spline_layer, Dense(W_out, b_out))
#
#     # Store the weights and biases
#     weights_store = nnbackend_finest.cfWeights
#     biases_store  = nnbackend_finest.cfBiases
#
#     # generate data x and ground truth
#     xdata = readdlm("$(@__DIR__)/data/peaks/features.dat")
#     ydata = readdlm("$(@__DIR__)/data/peaks/labels.dat")
#    perm = randperm(size(xdata,1))
#    xdata = xdata[perm,:]'
#    ydata = ydata[perm,:]'
#    x = xdata[:,1]
#
#    y_truth = model(x)
#    println("y_truth: ", y_truth)
#
#      # Iterate for coarser h
#      err_vec = zeros(0)
#      h_vec = zeros(0)
#      for i in 1:10
#
#         # Increase h
#         h = h * 10.0
#         nlayers = round(Int64, T/h)
#         # No less than 10 layers
#         if nlayers < 1
#             break
#         end
#
#         # Generate next coarser model
#         paras = SplineParameters{Float64,degree,typeof(myactivation)}(nLayers=nlayers, hLayers=h,
#                    nKnots=nknots, amplitude=amp, σ=myactivation)
#         nnbackend = DenseNetworkBackend(layer_dim, paras, batchSize=batch_size, init_λ=init_λ,
#                    init_weights_zeros=false, init_biases_zeros=false)
#         # Set the weights from the finest model
#         nnbackend.cfWeights = weights_store
#         nnbackend.cfBiases  = biases_store
#         spline_layer = SplineNetwork(paras, nnbackend)
#         model = Chain(Dense(W_in, b_in), spline_layer, Dense(W_out, b_out))
#
#         # Propagate forward and compute error
#         y = model(x)
#
#         println(y)
#         abs_err = norm(y - y_truth, 2)
#         println(h, "   ", abs_err)
#         append!(err_vec, abs_err)
#         append!(h_vec, h)
#
#     end
#
#     println("stepsize   = ", h_vec)
#     println("abs. error = ", err_vec)
#     return h_vec, err_vec
# end
#
#  #Do the delta t test for degree 2 and 3
#   h_vec_S2, err_vec_S2 = dttest(2);
#   # h_vec_S3, err_vec_S3 = dttest(3);
#   plot(h_vec_S2, err_vec_S2, yaxis=:log, xaxis=:log, marker=:d, legend=:bottomright, label="SpliNet, d=2", xlabel="h", ylabel="error of network output")
#   # plot!(h_vec_S3, err_vec_S3, yaxis=:log, xaxis=:log, marker=:d, legend=:bottomright, label="SpliNet, d=3", xlabel="h", ylabel="error of network output")
#   savefig("plots/peaks_dttest_T2_nknots20_randominit.png")
