using JLD, Statistics, Plots

## explore "smoothness" of weights/biases
function plot_weights(bson_filename)
    model = load(bson_filename)[:model][2]
    width = model.args.width
    cfWeights = model.cfWeights
    cfBiases = model.cfBiases
    display(plot())
    for i in 1:width
        for j in 1:width
            display(plot!(cfWeights[i,j,1:end-1], label=nothing))
        end
    end
    display(plot())
    for i in 1:width
        display(plot!(cfBiases[i,1:end-1], label=nothing))
    end
end

## save hyperparameter optimization results
function save_and_plot(network_type, degree, ho_param, poststring="")

    network_name = String(network_type)*string(degree)

    # Save best hyperparameters and hyperopt statistics
    best_param, best_acc = minimum(ho_param)
    dict_hyper = Dict("hyper_paras" => best_param)
    dict_params = Dict("params" => ho_param.history)
    dict_stats = Dict("stats" => ho_param.results)

    output_folder = "$(@__DIR__)/../examples/plots/"
    filename_dict_hyper = output_folder*"$(network_name)_hyperopt_bestparams$(poststring).jld"
    filename_dict_stats = output_folder*"$(network_name)_hyperopt_stats$(poststring).jld"
    filename_dict_params = output_folder*"$(network_name)_hyperopt_params$(poststring).jld"
    filename_accuracies_png = output_folder*"$(network_name)_hyperopt_accuracies$(poststring).png"

    # Save hyperoptimization results
    save(filename_dict_hyper, dict_hyper)
    save(filename_dict_stats, dict_stats)
    save(filename_dict_params, dict_params)

    # Plot histograms
    p1 = plot(ho_param, ylabel="accuracy", markersize=10, yaxis=:log)
    p2 = histogram(ho_param.results, xlabel="accuracy", label=nothing)
    param = string("max: ", string(Float32(maximum(ho_param.results))))
    param = string(param, "   min: ", string(Float32(minimum(ho_param.results))))
    param = string(param, "   mean: ", string(Float32(mean(ho_param.results))))
    param = string(param, "   std: ", string(Float32(std(ho_param.results))))
    p0 = scatter(ones(3), marker=0, markeralpha=0, annotations=(2, 1, Plots.text(param)),
                 axis=false, leg=false, ylims=(0.8,1.2))
    display(plot(p1, p2, p0, layout = @layout([a;b{0.4h};c{0.05h}]), size=(1200,1200)))
    savefig(filename_accuracies_png)

    # Screen output
    println("\n Hyperopt statistics: \n", param, "\n")

    # Screen output: Report on output files
    println("Files written: ")
    println(" ", filename_dict_hyper)
    println(" ", filename_dict_stats)
    println(" ", filename_accuracies_png)
    println("Call the 'stats_plot' function to create a box plot from the statistics in the '*_stats.jld' file.\n")
end


## Create boxplots
# Call this after hyperparameter tuning has finished.
# Reads hyperparameter optimization results from the specified file
#   "<networkname>_hyperopt_stats.jld"
using StatsPlots, DataFrames
function stats_plot(filename_hyperoptstats, network_name)
    # Load hyperopt file
    ho_stats = load(filename_hyperoptstats)["stats"]

    # create boxplot.
    cat = [network_name]
    cat = vcat(fill.(cat,[length(ho_stats)])...)
    results = vcat(ho_stats)
    data = DataFrame(cat=cat, acc=results)
    # @df data violin(:cat, :acc, marker=(0.2,:blue,stroke(0)), label=nothing, yaxis=:log)
    @df data boxplot(:cat, :acc, marker=(0.3,:orange,stroke(2)), alpha=0.75, label=nothing, yaxis=:log)
    @df data dotplot!(:cat, :acc, marker=(:black,stroke(0)), label=nothing, yaxis=:log)

    # Save the plot
    idx = findfirst(".jld", filename_hyperoptstats)
    output_filename = SubString(filename_hyperoptstats, 1:idx[1]-1)*"_boxplot.png"
    savefig(output_filename)
    println("File written: ", output_filename)
end


## obtain statistics of output
function plot_stats(ho_resnet, ho_spline)
    # Plot histograms
    p_r = histogram(ho_resnet, xlabel="resnet", ylabel="accuracy")
    p_s = histogram(ho_spline, xlabel="spline", ylabel="accuracy")

    p1 = string("max: ", string(Float32(maximum(ho_resnet))))
    p1 = string(p1, "   min: ", string(Float32(minimum(ho_resnet))))
    p1 = string(p1, "\n   mean: ", string(Float32(mean(ho_resnet))))
    p1 = string(p1, "   std: ", string(Float32(std(ho_resnet))))
    p1 = scatter(ones(3), marker=0, markeralpha=0, annotations=(2, 1, Plots.text(p1)),
                 axis=false, leg=false, ylims=(0.8,1.2))

     p2 = string("max: ", string(Float32(maximum(ho_spline))))
     p2 = string(p2, "   min: ", string(Float32(minimum(ho_spline))))
     p2 = string(p2, "\n   mean: ", string(Float32(mean(ho_spline))))
     p2 = string(p2, "   std: ", string(Float32(std(ho_spline))))
     p2 = scatter(ones(3), marker=0, markeralpha=0, annotations=(2, 1, Plots.text(p2)),
                  axis=false, leg=false, ylims=(0.8,1.2))
    display(plot(p_r, p_s, p1, p2, layout = @layout([a b;c d{0.1h}]), size=(1200,600)))
end


# casename = "sine"
#
# ho_resnet = load("../examples/results/$(casename)_resnet_hyperopt_stats.jld")["stats"]
# ho_spline = load("../examples/results/$(casename)_spline_hyperopt_stats.jld")["stats"]
#
# plot_stats(ho_resnet, ho_spline)
