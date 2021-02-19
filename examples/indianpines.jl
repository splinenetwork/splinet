using DelimitedFiles, LinearAlgebra, Plots, Zygote
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Data: DataLoader
using Base: @kwdef


include("$(@__DIR__)/../src/unified_spline_network.jl")
include("$(@__DIR__)/../src/train.jl")
# include("$(@__DIR__)/../src/hyperopt.jl")

# Read training data set
ntrain = 4000
xtrain = readdlm("$(@__DIR__)/data/indianpines/features_training_shuffle.dat")[1:ntrain,:]'
ytrain = readdlm("$(@__DIR__)/data/indianpines/labels_training_shuffle.dat")[1:ntrain,:]'

# Read test data set
ntest = 1000
xtest = readdlm("$(@__DIR__)/data/indianpines/features_validation_shuffle.dat")[1:ntest,:]'
ytest = readdlm("$(@__DIR__)/data/indianpines/labels_validation_shuffle.dat")[1:ntest,:]'

# Get cross validation data set
ncrossval = 1000
crossval_start = ntest+1
crossval_stop  = (crossval_start + ncrossval > 2000) ? 2000 : crossval_start + ncrossval
xval = readdlm("$(@__DIR__)/data/indianpines/features_validation_shuffle.dat")[crossval_start:crossval_stop,:]'
yval = readdlm("$(@__DIR__)/data/indianpines/labels_validation_shuffle.dat")[crossval_start:crossval_stop,:]'

# Read index mappings for plotting cross validation predictions
idmap_train_to_color  = readdlm("$(@__DIR__)/data/indianpines/idmap_val_to_color.dat", Int16)[crossval_start:crossval_stop]
idmap_color_to_global = readdlm("$(@__DIR__)/data/indianpines/idmap_color_to_global.dat", Int16)

# Specify dimension of input and output data
const data_dim_in = 220
const data_dim_out = 16
# data shape in 4d: length x width x height x channel
const data_dims = (220,1,1,1)

## Data loading function
function getdata(batchsize)
    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=batchsize, shuffle=false)
    test_data = DataLoader(xtest, ytest, batchsize=batchsize)

    return train_data, test_data
end


## Plot prediction
function plot_predictions(model)
    dimx = 145    # Global image dimension 145 x145
    nclasses = 16 # number of classes

    # Get prediction on cross validation set
    predicted_classes = onecold(model(xval))

    # Map to global image
    img = zeros(dimx*dimx)
    for k in 1:length(predicted_classes)

        # Map from data to global image
        k_color  = idmap_train_to_color[k]
        k_global = idmap_color_to_global[k_color]

        # Set the class value
        classid = predicted_classes[k]
        img[k_global] = classid
    end

    # Reshape to image size
    img_mat = reshape(img, (dimx,dimx))

    # Plot as heatmap
    x = range(1, dimx, step=1)
    heatmap(x, x, img_mat, seriescolor=cgrad(:magma, nclasses), aspect_ratio=:equal, legend=:none)
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
function train_network(network_type, deg; learning_rate=3e-3, epochs=150,
                batch_size=100, regularize=1e-6, layer_dim=data_dims, nlayers=60, stepsize=0.01,
                nknots=10, amplitude=1.0, init_lambda=1.0, gnorm_tol=1e-4, do_plot=true, verbose=true)

    # Specify activation function
    myactivation(x) = relu(x)

    # Specify loss function
    myloss(x,y) = logitcrossentropy(x,y)

    # Specify target accuracy, 1.0 (=100%) for classification
    target_accuracy = 1.0

    # Specify convolution parameters
    filDim = (3,1,1)
    stride = 1

    # Set up Hidden layers
    nKnots = (network_type == :spline) ? nknots : nlayers
    paras = SplineParameters{Float32,deg,typeof(myactivation)}(nLayers=nlayers, hLayers=stepsize,
                    nKnots=nknots, amplitude=amplitude, σ=myactivation)
    conv_paras = ConvParameters(filterDims=filDim, stride=stride)
    nnbackend = ConvNetworkBackend(paras, conv_paras; dataDims=layer_dim, batchSize = batch_size,
                    init_weights_zeros=false, init_biases_zeros=false, init_λ=init_lambda)
    spline_layer = SplineNetwork(paras, nnbackend)

    # Set up opening and cosing weights and biases
    W_in  = Float32(amplitude) .* rand(Float32, data_dim_in, data_dim_in)
    b_in  = Float32(amplitude) .* rand(Float32, data_dim_in)
    W_out = Float32(amplitude) .* rand(Float32, data_dim_out, data_dim_in)
    b_out = Float32(amplitude) .* rand(Float32, data_dim_out)

    layer_in = Chain(Dense(W_in, b_in), change_dim_in)
    layer_out = Chain(change_dim_out, Dense(W_out, b_out))
    # Set up network model: [Opening layer, Hidden layers, closing layer]
    model = Chain(layer_in, spline_layer, layer_out)

    # Training parameters
    args = TrainingParameters(η=learning_rate, batchsize=batch_size, epochs=epochs, λ=regularize)
    # Call a general training function
    final_accuracy_best = train(args; myloss=myloss, model=model, network_name=network_type,
                            target_accuracy=target_accuracy, gnorm_tol=gnorm_tol, do_plot=do_plot, verbose=verbose)

    return final_accuracy_best
end

## Train once
# train_network(:resnet, 1)
# train_network(:spline, 2)
# train_network(:spline, 2; epochs=1)
