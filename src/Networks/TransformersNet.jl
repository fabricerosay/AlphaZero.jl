struct RoPE{T<:AbstractFloat, A<:AbstractArray{T}}
    head_size::Int
    cos_cached::A
    sin_cached::A
end
Flux.@functor RoPE


function RoPE(head_size::Int, seq_len::Int; base=10_000, T=Float32)
    half = head_size ÷ 2
    inv_freq = T.(1 ./ (base .^ ((0:2:head_size-1) ./ head_size)))
    t = T.(0:seq_len-1)
    freqs = inv_freq * t' # (half, seq_len)
    
    # We interleave manually so the shape matches (head_size, seq_len)
    # This allows: x[1]*cos[1], x[2]*cos[2]... 
    full_freqs = zeros(T, head_size, seq_len)
    full_freqs[1:2:end, :] .= freqs
    full_freqs[2:2:end, :] .= freqs

    return RoPE(head_size, cos.(full_freqs), sin.(full_freqs))
end

function (rope::RoPE)(x::AbstractArray{T, 3}) where T
    h, s, b = size(x)
    
    # 1. On récupère les cos/sin pré-calculés pour la longueur de séquence s
    # On utilise l'indexation directe plutôt que view pour éviter les SubArrays complexes
    c = rope.cos_cached[:, 1:s]
    s_val = rope.sin_cached[:, 1:s]
    
    # 2. On sépare les composantes paires et impaires
    # Au lieu de views, on crée des nouveaux tableaux (Zygote préfère)
    x_even = x[1:2:end, :, :]
    x_odd  = x[2:2:end, :, :]
    
    c_even = c[1:2:end, :]
    s_even = s_val[1:2:end, :]

    # 3. Calcul de la rotation : (x1*cos - x2*sin) et (x2*cos + x1*sin)
    y_even = @. x_even * c_even - x_odd * s_even
    y_odd  = @. x_odd * c_even + x_even * s_even
    
    # 4. Reconstruction par entrelacement (Interleaving)
    # C'est l'étape critique. On concatène et on reshappe pour retrouver l'ordre
    # On crée une matrice (2, half, seq, batch) puis on aplatit les deux premières dims
    combined = vcat(reshape(y_even, 1, :, s, b), reshape(y_odd, 1, :, s, b))
    
    return reshape(combined, (h, s, b))
end


"""
    SimpleResNetHP

Hyperparameters for the simple Resnet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width::Int`                  | Number of neurons on each dense layer        |
| `depth_common::Int`           | Number of dense layers in the trunk          |
| `depth_phead::Int = 1`        | Number of hidden layers in the actions head  |
| `depth_vhead::Int = 1`        | Number of hidden layers in the value  head   |
"""
@kwdef struct RRTNetHP
    entry::Tuple{Int,Int,Int}
    head_size::Int
    num_heads::Int
end



"""
    SimpleResNet <: FluxNetwork

A simple two-headed architecture with only dense layers and skip connections.
"""
mutable struct RRTNet <: FluxNetwork
    hyper
    common
    vhead
    phead
    rope
end

Flux.@functor RRTNet (common, vhead, phead)

function make_res(n)
    return Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Conv((3,3),n=>n,pad=1),Flux.BatchNorm(n,Flux.relu),Flux.Conv((3,3),n=>n,pad=1),Flux.BatchNorm(n)), +), Flux.relu)
end

struct TransformerBlock
    w_q::Dense
    w_k::Dense
    w_v::Dense
    w_out::Dense
    rope::RoPE        # The Zygote-safe RoPE struct we built
    norm1::LayerNorm
    norm2::LayerNorm
    ffn::Chain
    nheads::Int
end

Flux.@functor TransformerBlock
Flux.trainable(t::TransformerBlock)=(;w_q=t.w_q,w_k=t.w_k,w_out=t.w_out,norm1=t.norm1,norm2=t.norm2,ffn=t.ffn)

function TransformerBlock(c::Int, nheads::Int, rope::RoPE)
    # Ensure head_dim is compatible with RoPE
    head_dim = c ÷ nheads
    return TransformerBlock(
        Dense(c, c),           # w_q
        Dense(c, c),           # w_k
        Dense(c, c),           # w_v
        Dense(c, c),           # w_out
        rope,
        LayerNorm(c),
        LayerNorm(c),
        Chain(Dense(c, 4c, relu), Dense(4c, c)),
        nheads
    )
end

function (t::TransformerBlock)(x::AbstractArray{T, 4}) where T
    # x: (W, H, Channels, Batch) -> (6, 7, 64, 8)
    w, h, c, b = size(x)
    seq_len = w * h
    
    # 1. Flatten spatial dims to sequence: (Channels, Seq, Batch)
    # We use permutedims to ensure spatial proximity is preserved
    flat_x = reshape(permutedims(x, (3, 1, 2, 4)), (c, seq_len, b))
    
    # 2. Linear Projections
    q = t.w_q(flat_x)
    k = t.w_k(flat_x)
    v = t.w_v(flat_x)
    
    # 3. Apply RoPE ONLY to Q and K
    # Our RoPE expects (head_dim, seq, batch * nheads)
    # So we reshape to split heads before rotating
    q_heads = reshape(q, (c ÷ t.nheads, t.nheads, seq_len, b))
    k_heads = reshape(k, (c ÷ t.nheads, t.nheads, seq_len, b))
    
    # Merge nheads into batch for the RoPE layer
    q_rope = t.rope(reshape(q_heads, (c ÷ t.nheads, seq_len, t.nheads * b)))
    k_rope = t.rope(reshape(k_heads, (c ÷ t.nheads, seq_len, t.nheads * b)))
    
    # 4. Scaled Dot-Product Attention
    # Restore head dimension for NNlib's attention
    q_final = reshape(q_rope, (c ÷ t.nheads, seq_len, t.nheads, b))
    k_final = reshape(k_rope, (c ÷ t.nheads, seq_len, t.nheads, b))
    v_final = reshape(v, (c ÷ t.nheads, seq_len, t.nheads, b))
    
    # attn_out: (head_dim, seq, nheads, batch)
    attn_out,_ = dot_product_attention(q_final, k_final, v_final)
    
    # 5. Residual + FFN (Standard Transformer steps)
    attn_merged = reshape(attn_out, (c, seq_len, b))
    h_mid = t.norm1(flat_x + t.w_out(attn_merged))
    h_final = t.norm2(h_mid + t.ffn(h_mid))
    
    # 6. Reshape back to Grid: (W, H, Channels, Batch)
    out = reshape(h_final, (c, w, h, b))
    return permutedims(out, (2, 3, 1, 4))
end

function RRTNet(outdim::Int, hyper::RRTNetHP)
    rng = MersenneTwister(3409)
    weight_init() = Flux.glorot_uniform(rng)
    rope=RoPE(div(hyper.head_size,4),hyper.entry[1]*hyper.entry[2])
    common = Flux.Chain(
        x->reshape(x,hyper.entry...,length(size(x))==1 ? 1 : size(x)[end]),
        Flux.Conv((3,3),hyper.entry[3]=>hyper.head_size,pad=1),
        Flux.BatchNorm(hyper.head_size,relu),
        make_res(hyper.head_size),
        make_res(hyper.head_size),
        TransformerBlock(hyper.head_size,4,rope),
        make_res(hyper.head_size),
        make_res(hyper.head_size),
       TransformerBlock(hyper.head_size,4,rope),
        make_res(hyper.head_size),
        make_res(hyper.head_size),
        TransformerBlock(hyper.head_size,4,rope),
        make_res(hyper.head_size),
        make_res(hyper.head_size),
    )

    vhead = Flux.Chain(
        Flux.Conv((1,1),hyper.head_size=>32,pad=0),
        Flux.BatchNorm(32,relu),
        Flux.flatten,
        Flux.Dense(hyper.entry[1]*hyper.entry[2]*32=>1, tanh; init=weight_init())
    )

    phead = Flux.Chain(
        Flux.Conv((1,1),hyper.head_size=>32,pad=0),
        Flux.BatchNorm(32,relu),
        Flux.flatten,
        Flux.Dense(hyper.entry[1]*hyper.entry[2]*32=>outdim; init=weight_init())
    )

    RRTNet(hyper, common, vhead, phead,rope)
end


HyperParams(::Type{RRTNet}) = RRTNetHP
hyperparams(nn::RRTNet) = nn.hyper
on_gpu(nn::RRTNet) = arr_is_on_gpu(nn.vhead[end].bias)


function forward(nn::RRTNet, states, use_softmax=false)
    common = nn.common(states)
    v = nn.vhead(common)
    p = nn.phead(common)
    use_softmax && (p = Flux.softmax(p))
    return v, p
end