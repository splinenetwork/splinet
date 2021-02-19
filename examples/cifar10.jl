using Flux, LinearAlgebra, Plots, Statistics, Metalhead, Random, Zygote
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Data: DataLoader
using Parameters: @with_kw
using Base: @kwdef
using Metalhead: trainimgs
using Images: channelview
using Base.Iterators: partition


include("$(@__DIR__)/../src/unified_spline_network.jl")
include("$(@__DIR__)/../src/train.jl")
# include("$(@__DIR__)/../src/hyperopt.jl")

## Data loading functions
# Function to convert the RGB image to Float32 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(splitrate)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    ntrain = Int(splitrate * 50000)
    perm = randperm(ntrain)

    xtrain_arrays = [getarray(X[i].img) for i in perm]

    xtrain = zeros(Float32,32,32,3,ntrain)
    for i in 1:ntrain
        xtrain[:,:,:,i] .= xtrain_arrays[i]
    end
    # xtrain = cat(xtrain..., dims = 4)
    #onehot encode labels of batch
    ytrain = onehotbatch([X[i].ground_truth.class for i in perm],1:10)

    return xtrain, ytrain
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)
    ntest = 10000

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    xtest_arrays = [getarray(test[i].img) for i in 1:ntest]
    xtest = zeros(Float32,32,32,3,ntest)
    for i in 1:ntest
        xtest[:,:,:,i] = xtest_arrays[i]
    end
    # xtest = cat(xtest..., dims = 4)
    ytest = onehotbatch([test[i].ground_truth.class for i in 1:ntest], 1:10)

    return xtest, ytest
end

function getdata(batchsize)
    xtrain, ytrain = get_processed_data(1.0)
    xtest, ytest = get_test_data()
    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=false)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize)

    return train_data, test_data
end

## Specify dimension of input and output data
#  data shape in 4d: length x width x height x channel
const chans = 16
const data_dims = (32,32,1,16)
const data_dim_out = 10

## Plot prediction
#   CIFAR10 has no easy prediction visualization!!
function plot_predictions(model)
    plot(bg=:white)
end

## Compute accuracy of the model
function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(model(x)) .== onecold(y)) * 1 / size(x)[end]
    end
    acc/length(data_loader)
end

change_dim_in(x::AbstractArray) = reshape(x, data_dims..., size(x)[end])
Zygote.@adjoint change_dim_in(x::AbstractArray) = change_dim_in(x), z -> (reshape(z,size(x)...), )

change_dim_out(x::AbstractArray) = reshape(x, prod(data_dims), size(x)[end])
Zygote.@adjoint change_dim_out(x::AbstractArray) = change_dim_out(x), z -> (reshape(z,size(x)...),)



## Set up test case and call main training function
# This function is used by hyperopt.jl for tuning hyperparameters
function train_network(network_type, deg; learning_rate=1e-2, epochs=40,
                batch_size=100, regularize=1e-4, layer_dim=data_dims, nlayers=10, stepsize=0.1,
                nknots=5, amplitude=1.0, init_lambda=1.0, gnorm_tol=1e-4, do_plot=true, verbose=true)

    # Specify activation function
    myactivation(x) = relu(x)

    # Specify loss function
    myloss(x,y) = logitcrossentropy(x,y)

    # Specify target accuracy, 1.0 (=100%) for classification
    target_accuracy = 1.0

    conv_in = Conv((3,3), 3 => chans, relu, pad=(1,1), stride=(1,1))

    model = Chain(conv_in, change_dim_in)

    # Specify convolution parameters
    filDim = (3,3,1)
    stride = 1
    # chan_in = 32
    # chan_out = 32

    layer_dim = (32,32,1,chans)

    # build 3 blocks with SpliNet + MaxPool
    for i in 1:3
        # Set up Hidden layers
        nKnots = (network_type == :spline) ? nknots : nlayers
        paras = SplineParameters{Float32,deg,typeof(myactivation)}(nLayers=nlayers, hLayers=stepsize,
                        nKnots=nknots, amplitude=amplitude, σ=myactivation)
        conv_paras = ConvParameters(filterDims=filDim, stride=stride, chan_in=chans, chan_out=chans)
        nnbackend = ConvNetworkBackend(paras, conv_paras; dataDims=layer_dim, batchSize = batch_size,
                        init_weights_zeros=false, init_biases_zeros=false, init_λ=init_lambda)
        spline_layer = SplineNetwork(paras, nnbackend)

        model = Chain(model..., spline_layer, MaxPool((2,2,1)))

        layer_dim = (Int(layer_dim[1]/2), Int(layer_dim[2]/2), layer_dim[3], chans)
    end

    model = Chain(model...,
                  MeanPool(layer_dim[1:3]),
                  flatten,
                  Dense(chans,data_dim_out))

    #
    # # Set up opening and cosing weights and biases
    # W_out = Float32(amplitude) .* rand(Float32, data_dim_out, prod(data_dims))
    # b_out = Float32(amplitude) .* rand(Float32, data_dim_out)
    #
    # # layer_in = Chain(Dense(W_in, b_in), change_dim_in)
    # layer_out = Chain(change_dim_out, Dense(W_out, b_out))
    # # Set up network model: [Opening layer, Hidden layers, closing layer]
    # model = Chain(change_dim_in, spline_layer, layer_out)


    # Training parameters
    args = TrainingParameters(η=learning_rate, batchsize=batch_size, epochs=epochs, λ=regularize)
    # Call a general training function
    final_accuracy_best = train(args; myloss=myloss, model=model, network_name=network_type,
                            target_accuracy=target_accuracy, gnorm_tol=gnorm_tol, do_plot=do_plot, verbose=verbose)

    return final_accuracy_best
end

## Train once
# @time train_network(:resnet, 1)
# train_network(:spline, 2)
# train_network(:spline, 2; epochs=1)
