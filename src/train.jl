using Flux
using JLD
using BSON: @save, @load
using Base: @kwdef

## Main training functionality
# To use this module, one should implement the following functions for each new test-case:
#
# * getdata(batchsize) -> return DataLoader train_data and test_data
# * plot_prediction(model): 2D plot visualizing model predictions
# * Optional: Overwrite the accuracy(data_loader, model) function to compute the accuracy of the model (e.g. if classification case, compute the percentage of correctly classified samples)
#
# For a new test case, set up the model (network including opening layer, spline layer, closing layer), specify loss, then call train(...)

## Training-parameters
@kwdef struct TrainingParameters
    η::Float64 = 3e-3       # learning rate
    batchsize::Int = 10     # batch size
    epochs::Int = 20        # number of epochs
    λ::Float64 = 1e-9       # weight of regularization
end

## Main training function
function train(args::TrainingParameters; model, myloss, network_name, target_accuracy=1.0,
            gnorm_tol=1e-4, do_plot=true, verbose=true) where {NT,D}

    # Load Data
    train_data, test_data = getdata(args.batchsize)

    # Initialize the optimizer
    opt = ADAM(args.η)

    # Initialize output and plots
    train_accuracy = zeros(args.epochs,)
    test_accuracy = zeros(args.epochs,)
    gd_norm = zeros(args.epochs,)
    train_loss = zeros(args.epochs,)
    test_loss = zeros(args.epochs,)
    plot_interval = 5

    # Initialize best model and test accuracy
    final_accuracy_best = 1.0 - target_accuracy
    # model_best = model  # requires a deepcopy!
    # λ_best = model[2].networkbackend.λ[1]

    # main training loop
    for epoch in 1:args.epochs
        if verbose
            println("epoch ", epoch)
        end

        # Train one epoch using custom training loop
        train_accuracy[epoch], test_accuracy[epoch], gd_norm[epoch] =
            custom_train!(myloss, args.λ, model, train_data, opt, test_data, do_plot, verbose)
        train_loss[epoch] = loss_all(train_data, myloss, model, args.λ)
        test_loss[epoch] = loss_all(test_data, myloss, model, args.λ)

        # Plot intermediate prediction and accuracy
        if do_plot && ( mod1(epoch, plot_interval) == 1 || epoch == args.epochs )
            display(plot_predictions_and_accuracy(args, model, 1:epoch, train_accuracy, test_accuracy,
                gd_norm, train_loss, test_loss, target_accuracy))
        end

        # Store best accuracy and model
        if abs(test_accuracy[epoch] - target_accuracy) < abs(final_accuracy_best - target_accuracy)
            final_accuracy_best = test_accuracy[epoch]
            # model_best = model  # requires a deepcopy!
            # λ_best = model[2].networkbackend.λ[1]
        end

        # Stopping criterion
        if gd_norm[epoch] < gnorm_tol
            break
        end
    end

    # Plot results & save parameters
    # if do_plot
    #     suffix = network_name
    #     savefig("$(@__DIR__)/../examples/plots/$(suffix)_net_results.png")
    #     dict_training = Dict("training_paras" => args)
    #     save("$(@__DIR__)/../examples/plots/$(suffix)_training_params.jld", dict_training)
    #     # Save best model
    #     # @save "$(@__DIR__)/../examples/plots/$(suffix)_model.bson" model_best
    # end
        # print best lambda
    # println("best time-scale: ", λ_best)

    return final_accuracy_best
end

## custom train loop
function custom_train!(myloss, regul, model, train_data, optimizer, test_data, do_plot, verbose=true)

    # Set up loss function with regularization
    loss(x,y) = myloss(model(x),y) + regul * mean(norm, Flux.params(model))

    ps = Flux.params(model)
    gd_norm = 0.0
    # loop over batched data to
    for d in train_data
        gs = gradient(() -> loss(d[1], d[2]), ps)
        Flux.update!(optimizer, ps, gs)
        for parameter in ps
            gd_norm += norm(gs[parameter], 2)
        end
    end
    gd_norm /= length(train_data)

    if verbose
        @show train_accuracy = accuracy(train_data, model)
        @show test_accuracy = accuracy(test_data, model)
    else
        train_accuracy = accuracy(train_data, model)
        test_accuracy = accuracy(test_data, model)
    end

    return train_accuracy, test_accuracy, gd_norm
end

## loss function & accuracy computation
function loss_all(dataloader, myloss, model, λ)
    l = 0f0
    for (x,y) in dataloader
        l += myloss(model(x), y)
    end
    l = l / length(dataloader)
    l += λ * mean(norm, Flux.params(model))
end

function accuracy end # DUMMY FUNCTION. Should be overwritten for each test case

## Plotting
function plot_predictions end # DUMMY FUNCTION. Should be overwritten for each test case

function plot_accuracy(idx, train_accuracy, test_accuracy, target_accuracy)
    epochs = length(train_accuracy)
    if target_accuracy == 0.0
        plot(
            idx,
            [train_accuracy[idx], test_accuracy[idx]],
            xlabel="epoch",
            label=["training accuracy" "validation accuracy"],
            legendfontsize=10,
            legend=:bottomright,
            xlims=(1,epochs),
            yaxis=:log
        )
    else
         plot(
            idx,
            [train_accuracy[idx], test_accuracy[idx]],
            xlabel="epoch",
            label=["training accuracy" "validation accuracy"],
            legendfontsize=10,
            legend=:bottomright,
            xlims=(1,epochs)
          )
    end
end

function plot_gradient_norm(idx, gd_norm)
    epochs = length(gd_norm)
    plot(
        idx,
        gd_norm[idx],
        xlabel="epoch",
        label="gradient norm",
        legendfontsize=10,
        xlims=(1,epochs),
        yaxis=:log
    )
end

function plot_loss(idx, train_loss, test_loss)
    epochs = length(train_loss)
    plot(
        idx,
        [train_loss[idx], test_loss[idx]],
        xlabel="epoch",
        label=["training loss" "validation loss"],
        legendfontsize=10,
        xlims=(1,epochs),
        yaxis=:log
    )
end

function plot_predictions_and_accuracy(training_param, model, idx, train_accuracy, test_accuracy,
                    gd_norm, train_loss, test_loss, target_accuracy)
    network_param = model[2].spArgs
    nnbackend = model[2].networkbackend
    dim_string = (ndims(nnbackend.xInterm) == 2) ? "$(nnbackend.width)" : "$(nnbackend.dataDims)"
    param = "degree: $(degree(network_param))" *
            "   learning rate: $(training_param.η)" *
            "   batch size: $(training_param.batchsize)" *
            "   regularization: $(training_param.λ) \n" *
            "   hidden layer dim: " * dim_string *
            "   layers: $(network_param.nLayers)" *
            "   h: $(network_param.hLayers)" *
            "   knots: $(network_param.nKnots)" *
            "   initial amplitude: $(network_param.amplitude) \n"
    p1 = plot_predictions(model)
    p2 = plot_accuracy(idx, train_accuracy, test_accuracy, target_accuracy)
    p3 = plot_gradient_norm(idx, gd_norm)
    p4 = plot_loss(idx, train_loss, test_loss)
    p0 = scatter(ones(3), marker=0, markeralpha=0, annotations=(2, 1, Plots.text(param)), axis=false, leg=false, grid=false, ylims=(0.8,1.2))
    plot(p1, p2, p3, p4, p0, layout = @layout([grid(2,2);a{0.1h}]), size=(800,800))
end
