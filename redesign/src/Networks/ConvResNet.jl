"""
    ConvResNetHP

Hyperparameters for the simple Resnet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width::Int`                  | Number of neurons on each dense layer        |
| `depth_common::Int`           | Number of dense layers in the trunk          |
| `depth_phead::Int = 1`        | Number of hidden layers in the actions head  |
| `depth_vhead::Int = 1`        | Number of hidden layers in the value  head   |
"""
@kwdef struct ConvResNetHP
    entry::Tuple{Int,Int,Int}
    hidden_size::Int
    depth_common::Int
end


"""
    ConvResNet <: FluxNetwork

A simple two-headed architecture with only dense layers and skip connections.
"""
mutable struct ConvResNet <: FluxNetwork
    hyper
    common
    vhead
    phead
end
Flux.@functor ConvResNet (common, vhead, phead)


function ConvResNet(outdim::Int, hyper::ConvResNetHP)
    rng = MersenneTwister(3409)
    weight_init() = Flux.glorot_uniform(rng)

    function make_res(n)
        return Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Conv((3,3),n=>n,pad=1),Flux.BatchNorm(n,Flux.relu),Flux.Conv((3,3),n=>n,pad=1),Flux.BatchNorm(n)), +), Flux.relu)
    end

    hidden_skip_layers(depth) = [make_res(hyper.hidden_size) for _ in 1:depth]

    common = Flux.Chain(
        x->reshape(x,hyper.entry...,length(size(x))==1 ? 1 : size(x)[end]),
        Flux.Conv((3,3),hyper.entry[3]=>hyper.hidden_size,pad=1),
        Flux.BatchNorm(hyper.hidden_size,relu),
        hidden_skip_layers(hyper.depth_common)...
    )

    vhead = Flux.Chain(
        Flux.Conv((1,1),hyper.hidden_size=>32,pad=0),
        Flux.BatchNorm(32,relu),
        Flux.flatten,
        Flux.Dense(hyper.entry[1]*hyper.entry[2]*32=>1, tanh; init=weight_init())
    )

    phead = Flux.Chain(
        Flux.Conv((1,1),hyper.hidden_size=>32,pad=0),
        Flux.BatchNorm(32,relu),
        Flux.flatten,
        Flux.Dense(hyper.entry[1]*hyper.entry[2]*32=>outdim; init=weight_init())
    )

    ConvResNet(hyper, common, vhead, phead)
end


HyperParams(::Type{ConvResNet}) = ConvResNetHP
hyperparams(nn::ConvResNet) = nn.hyper
on_gpu(nn::ConvResNet) = arr_is_on_gpu(nn.vhead[end].bias)


function forward(nn::ConvResNet, states, use_softmax=false)
    common = nn.common(states)
    v = nn.vhead(common)
    p = nn.phead(common)
    use_softmax && (p = Flux.softmax(p))
    return v, p
end


# @kwdef struct ResNetHP
#     num_blocks :: Int
#     num_filters :: Int
#     conv_kernel_size :: Tuple{Int, Int}
#     num_policy_head_filters :: Int = 2
#     num_value_head_filters :: Int = 1
#     batch_norm_momentum :: Float32 = 0.6f0
#   end
  
#   """
#       ResNet <: TwoHeadNetwork
  
#   The convolutional residual network architecture that is used
#   in the original AlphaGo Zero paper.
#   """

#   @kwdef struct Gspec
#     indim::Tuple{Int,Int,Int}=(6,7,2)
#     oudim::Int=7
#   end

#   mutable struct ResNet <: TwoHeadNetwork
#     gspec
#     hyper
#     common
#     vhead
#     phead
#   end
  
#   function ResNetBlock(size, n, bnmom)
#     pad = size .÷ 2
#     layers = Chain(
#       Conv(size, n=>n, pad=pad),
#       BatchNorm(n, relu, momentum=bnmom),
#       Conv(size, n=>n, pad=pad),
#       BatchNorm(n, momentum=bnmom))
#     return Chain(
#       SkipConnection(layers, +),
#       x -> relu.(x))
#   end
  
#   function ResNet(gspec::Gspec,hyper::ResNetHP)
#     indim = gspec.indim
#     outdim = gspec.outdim
#     ksize = hyper.conv_kernel_size
#     @assert all(ksize .% 2 .== 1)
#     pad = ksize .÷ 2
#     nf = hyper.num_filters
#     npf = hyper.num_policy_head_filters
#     nvf = hyper.num_value_head_filters
#     bnmom = hyper.batch_norm_momentum
#     common = Chain(
#       Conv(ksize, indim[3]=>nf, pad=pad),
#       BatchNorm(nf, relu, momentum=bnmom),
#       [ResNetBlock(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
#     phead = Chain(
#       Conv((1, 1), nf=>npf),
#       BatchNorm(npf, relu, momentum=bnmom),
#       flatten,
#       Dense(indim[1] * indim[2] * npf, outdim),
#       softmax)
#     vhead = Chain(
#       Conv((1, 1), nf=>nvf),
#       BatchNorm(nvf, relu, momentum=bnmom),
#       flatten,
#       Dense(indim[1] * indim[2] * nvf, nf, relu),
#       Dense(nf, 1, tanh))
#     ResNet((6,7,2), hyper, common, vhead, phead)
#   end
  
#   Network.HyperParams(::Type{<:ResNet}) = ResNetHP
  
  function Base.copy(nn::ConvResNet)
    return ConvResNet(
      nn.hyper,
      deepcopy(nn.common),
      deepcopy(nn.vhead),
      deepcopy(nn.phead)
    )
  end