# Spline Parameterized Neural Network
This repository contains a new type of deep neural network which are parameterized by B-spline basis functions, named SpliNet. 
The interpertation of ResNet as an numerical discretization of a continuous optimal control problem allows us to 
decouple the parameterization from the numerical scheme.

## Content
- src
  - unified_spline_network.jl
  - train.jl
  - hyperopt.jl
- examples
  - sine.jl
  - peaks.jl
  - indianpines.jl
  - cifar10.jl
  
## Source files
- unified_spline_network.jl  
  A unifided neural network struct which can deal with either vector or 2/3D tensor inputs. 
  For vector input, the network uses a dense matrix as the linear transformation while for 2/3D inputs, it uses a convolution filter.
  
- train.jl
  A customized function for training SpliNet allowing users to choose learning rate, batch size, epoch, regularization scale, target accuracy/error and so on.
  
- hyperopt.jl
  A hyper-parameter sampling and tuning function. 
  
## Julia
To run the code, Julia (v1.0 or later) needs to be installed (https://julialang.org/downloads/).   
To construct the network and run back-propagation, Flux and Zygote are needed, which can be installed by running
`] add Flux/Zygote` in Julia's REPL

## Run examples
To run the examples, change parameters in the desiring task under `examples` folder and run `include("*.jl")`.  
For the "sine" examples, you probably obtain the following visualization:

![plot](https://github.com/splinenetwork/splinet/blob/master/examples/plots/sine_for_readme.png)
