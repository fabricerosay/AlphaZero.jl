"""
    SimpleNetHP

Hyperparameters for the simplenet architecture.

| Parameter                              | Description                                  |
|:---------------------------------------|:---------------------------------------------|
| `width::Int`                           | Number of neurons on each dense layer        |
| `depth_common::Int`                    | Number of dense layers in the trunk          |
| `depth_phead::Int = 1`                 | Number of hidden layers in the actions head  |
| `depth_vhead::Int = 1`                 | Number of hidden layers in the value  head   |
| `use_batch_norm::Bool = false`         | Use batch normalization between each layer   |
| `batch_norm_momentum::Float32 = 0.6f0` | Momentum of batch norm statistics updates    |
"""
@kwdef struct NnueNetHP
  width::Int
end

function screlu(x)
    xp=relu(x)
    return min(one(x),xp*xp)
end
"""
    SimpleNet <: FluxNetwork

A simple two-headed architecture with only dense layers.
"""
mutable struct NnueNet <: FluxNetwork
    hyper
   common
end
Flux.@functor NnueNet (common,)


function NnueNet(hyper::NnueNetHP)
    rng = MersenneTwister(3409)
    weight_init() = Flux.glorot_uniform(rng)
     
    common = Flux.Chain(
        Flux.Dense(126 => hyper.width, screlu; init=weight_init()),
        Flux.Dense(2*hyper.width=>1,tanh;init=weight_init())
    )

  

    NnueNet(hyper,common)
end
function clamp_nn(nn::NnueNet)
    nn.common[1].weight.=clamp.(nn.common[1].weight,-1.0f0,1.0f0)
end

HyperParams(::Type{NnueNet}) = NnueNetHP
hyperparams(nn::NnueNet) = nn.hyper
on_gpu(nn::NnueNet) = arr_is_on_gpu(nn.common[1].bias)


function forward(nn::NnueNet, x, use_softmax=false)
    N=size(x,3)
    y=reshape(nn.common[1](x),:,N)
    return  nn.common[2](y)   
end

function rescale(nn::NnueNet,τ=1.2f0)
    println("rescaling...")
    nn.phead[end].weight./=τ
    nn.phead[end].bias./=τ
end

