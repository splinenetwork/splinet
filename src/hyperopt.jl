using JLD, Statistics, Plots
using Distributed
include("$(@__DIR__)/../src/util.jl")

# Add multiple processes
rmprocs(workers()...)
addprocs(4)

@everywhere using Hyperopt

## Hyperparameter optimization
# To use this module, one needs to implement the train_network function for a specific test case;
# train_network should take (named)variables as:
#   train_network(network_type, deg; learning_rate, epochs, batch_size, regularize,
#                 network_width, nlayers, stepsize, nknots, amplitude, do_plot, verbose)
# train_network should be included here; default test cases for this library contains:
# sine.jl   peaks.jl    indianpines.jl  cifar10.jl

@everywhere casename = "indianpines"
@everywhere include("$(@__DIR__)/../examples/$(casename).jl")

function get_layer_dimension(casename)
    if casename == "sine"
        layer_dim = 4
    elseif casename == "peaks"
        layer_dim = 5
    elseif casename == "indianpines"
        layer_dim = (220,1,1,1)
    elseif casename == "cifar10"
        layer_dim = (32,32,1,3)
    else
        print("Unknown data type!")
        layer_dim = 0
    end
    return layer_dim
end

## Tune a resnet
# This is standard resnet with fixed stepsize=1.0. Don't learn init_λ for this one! Disable it by removing it from Flux.trainable(...) constructor in 'unified_spline_network' before running this.
function resnet_hyperparam_tune(; nsamples=50, epochs=200, batch_size=20, do_plot=false, layer_dim)
    # Fixed parameters
    network_type = :resnet
    deg = 1
    init_λ= 1.0
    stepsize = 1.0

    # Hyperparameter tuning
    ho_param = @phyperopt for i=nsamples, sampler=CLHSampler(dims=[Continuous(),
                                    Continuous(), Categorical(6), Continuous()]),
                    learning_rate = exp10.(LinRange(-3, -1, nsamples)),
                    regularize = exp10.(LinRange(-10, -4, nsamples)),
                    nlayers = [2, 5, 10, 15, 20, 25],
                    amplitude = exp10.(LinRange(-3, 0, nsamples))
        nknots = nlayers  # RESNET!
        print_params(i, learning_rate, regularize, nlayers, nknots, layer_dim, stepsize, amplitude)
        @show train_network(network_type, deg; learning_rate=learning_rate, epochs=epochs,
                    batch_size=batch_size, regularize=regularize, layer_dim=layer_dim,
                    nlayers=nlayers, stepsize=stepsize, nknots=nknots, amplitude=amplitude,
                    init_λ=init_λ, do_plot=false, verbose=false)
    end


    if do_plot
        poststring = ""
        save_and_plot(network_type, deg, poststring, ho_param)
    end
    return ho_param
end

## Tune a SpliNet
# This can be either a true SpliNet (deg >=2, nknots <= nlayers) or a standard ODEnet (deg=1, nknots=nlayers)
function spline_hyperparam_tune(; deg=2, nsamples=50, epochs=200, batch_size = 20, T=1.0, stepsize=0.01, init_lambda=1.0, do_plot=false, layer_dim)

    # Fixed parameters
    network_type = :spline

    # Hyperparameter tuning
    ho_param = @phyperopt for i=nsamples, sampler=CLHSampler(dims=[Continuous(),
                                    Continuous(), Categorical(6), Continuous()]),
                    learning_rate = exp10.(LinRange(-3, -1, nsamples)),
                    regularize = exp10.(LinRange(-10, -4, nsamples)),
                    nknots = [2, 5, 10, 15, 20, 25],
                    amplitude = exp10.(LinRange(-3, 0, nsamples))
        # If standard ODEnet, nknots=nlayers and overwrite stepsize to be T/nlayers
        if deg == 1
            nlayers = nknots
            stepsize = T / nlayers
        else # SpliNet, fixed stepsize (-> fixed nlayers)
            nlayers = round(Int64, T/stepsize)
        end
        print_params(i, learning_rate, regularize, nlayers, nknots, layer_dim, stepsize, amplitude)
        @show train_network(network_type, deg; learning_rate=learning_rate, epochs=epochs,
                        batch_size=batch_size, regularize=regularize, layer_dim=layer_dim,
                        nlayers=nlayers, stepsize=stepsize, nknots=nknots, amplitude=amplitude,
                        init_λ=init_lambda, do_plot=false, verbose=false)
    end


    if do_plot
        poststring = "_initlambda"*string(init_lambda)
        save_and_plot(network_type, deg, poststring, ho_param)
    end

    return ho_param
end

## helper function : print current parameters
@everywhere function print_params(i, learning_rate, regularize, nlayers, nknots,
                            layer_dim, stepsize, amplitude)
    print("Training ", i, "  learning rate: ", learning_rate,  "  regularization: ", regularize,
        "  nlayers: ", nlayers, "  nknots: ", nknots, "  width: ", layer_dim, "  stepsize: ", stepsize,
        "  amplitude: ", amplitude, "\n")
end

## run parallel tests on resnet & splinet
function main_hyperopt(; n=100, init_lambda=1.0)
    layer_dim = get_layer_dimension(casename)

    # standard Resnet: Do not learn λ! Remove λ from trainable parameters at the end of file unified_spline_network.jl!
    # ho_resnet = resnet_hyperparam_tune(; nsamples=n, epochs=100, layer_dim=layer_dim)

    # Tune SpliNet, degree=2
    ho_spline = spline_hyperparam_tune(; deg=2, nsamples=n, epochs=1000, batch_size=5, init_lambda=init_lambda, layer_dim=layer_dim)

    # Tune ODEnet
    ho_odenet = spline_hyperparam_tune(; deg=1, nsamples=n, epochs=1000, batch_size=5, init_lambda=init_lambda, layer_dim=layer_dim)

    # resnet_result = Dict("stats" => ho_resnet.results)
    spline_result = Dict("stats" => ho_spline.results)
    odenet_result = Dict("stats" => ho_odenet.results)

    # save("$(@__DIR__)/../examples/plots/$(casename)_resnet_hyperopt_stats.jld", resnet_result)
    save("$(@__DIR__)/../examples/plots/$(casename)_spline_hyperopt_stats.jld", spline_result)
    save("$(@__DIR__)/../examples/plots/$(casename)_odenet_hyperopt_stats.jld", odenet_result)

end

# @time main_hyperopt(n=2, init_λ=1.0)
