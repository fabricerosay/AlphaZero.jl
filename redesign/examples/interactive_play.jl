using RLZero.BatchedMcts
using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Tests.Common.BitwiseConnectFour
using RLZero.TrainUtilities: init_mcts_config
using RLZero.Util.Devices
using RLZero
using Flux
using JLD2

const MCTS = BatchedMcts

# set these constants to your preference
const DEVICE = CPU()
const MODEL_PATH = "examples/models_nnue/connect-four-checkpoints/model_007700.jld2"
const nn_config = SimpleNetHP(width=192, depth_common=1)


function load_nn()
    state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
    action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)
    nn = SimpleNet(state_dim..., action_dim, nn_config)

    model_state = JLD2.load(MODEL_PATH, "model_state");
    Flux.loadmodel!(nn, model_state);
    nn = (DEVICE == CPU()) ? Flux.cpu(nn) : Flux.gpu(nn)
    return nn
end


function load_nnue(number)
    nnue_hyper_params=NnueNetHP(832)
    nnue_cpu=NnueNet(nnue_hyper_params)
    model_state = JLD2.load("examples/models_nnue/connect-four-checkpoints/model_$number.jld2", "model_state");
    Flux.loadmodel!(nnue_cpu, model_state);
    reseau=nnue_cpu.common
    adapt_weights(reseau,832)
    return reseau
end

function _mcts_action(env, mcts_config)
    env_vec = DeviceArray(mcts_config.device)([env])
    println("Thinking..")
    tree = MCTS.explore(mcts_config, env_vec)
    action = Array(MCTS.evaluation_policy(tree, mcts_config))[1]
    println("AlphaZero chooses action: $action\n\n", " value :", RLZero.BatchedMcts.value(tree|>cpu,1,1))
    return action
end

function _nn_move(env, nn)
    na = BatchedEnvs.num_actions(typeof(env))
    state = BatchedEnvs.vectorize_state(env)
    state = Flux.unsqueeze(state, length(size(state)) + 1)
    on_gpu(nn) && (state = DeviceArray(DEVICE)(state))
    _, logits = forward(nn, state)
    logits = Array(logits)[:, 1]
    invalid_actions = [!BatchedEnvs.valid_action(env, a) for a in 1:na]
    logits[invalid_actions] .= -Inf
    action = argmax(logits)
    println("Raw Neural Network chooses action: $action\n\n")
    return action
end

function _parse_player_action(env)
    na = BatchedEnvs.num_actions(typeof(env))
    valid_actions = findall(!iszero, [BatchedEnvs.valid_action(env, a) for a in 1:na])
    print("Input action: ")
    action = tryparse(Int16, readline())
    while isnothing(action) || action ∉ valid_actions
        print("Invalid action. Please choose one of $valid_actions. Input action: ")
        action = tryparse(Int16, readline())
    end
    println("User chooses action: $action\n\n")
    return action
end

"""Play against an MCTS player using the provided neural network."""

posidx(n, player) = n + (63) * player
posidx(x, y, player) = posidx(9 * (x - 1) + y, player)
function get_action_id(env::BitwiseConnectFourEnv, action)
    at(i, j, player) = env.board[posidx(i, j, player)]

    curr_row = 7
    while at(curr_row, action, env.curplayer) || at(curr_row, action, !env.curplayer)
        curr_row -= 1
    end
  
    return  posidx(curr_row, action, env.curplayer)%63
   
end
function play_with_mcts(nn, mcts_kwargs, az_goes_first = true)
    mcts_config = init_mcts_config(DEVICE, nn, mcts_kwargs)
   
    env = BitwiseConnectFourEnv()
    env,_=BatchedEnvs.act(env, 8)
    println(env)
    
    az_plays = az_goes_first
    done, info = false, nothing
    while !done
        
        t=@elapsed action =az_plays ? _mcts_action(env, mcts_config) : _parse_player_action(env)
        println("time: $t")
        println("action id $(get_action_id(env,action)-1)")
        env, info = BatchedEnvs.act(env, action)
        info.switched && (az_plays = !az_plays)
        done = BatchedEnvs.terminated(env)
        println(env)
    end

    info.switched && (az_plays = !az_plays)
    last_player = az_plays ? "AlphaZero" : "User"
    println("Game terminated! Last reward: $(info.reward) by player: $last_player.")
end

"""Play against the greedy policy of the neural network."""
function play_with_nn(nn, nn_goes_first)
    env = BitwiseConnectFourEnv()
    println(env)

    nn_plays = nn_goes_first
    done, info = false, nothing
    while !done
        action = nn_plays ? _nn_move(env, nn) : _parse_player_action(env)
        env, info = BatchedEnvs.act(env, action)
        info.switched && (nn_plays = !nn_plays)
        done = BatchedEnvs.terminated(env)
        println(env)
    end

    info.switched && (nn_plays = !nn_plays)
    last_player = nn_plays ? "Neural Network" : "User"
    println("Game terminated! Last reward: $(info.reward) by player: $last_player.")
end


# load the neural network
nn = load_nn()

# # set here the important values for the MCTS as a NamedTuple
mcts_kwargs = (;
    # common MCTS variables
    use_gumbel_mcts = false,
    num_simulations = 20000,

    # Gumbel MCTS variables -- No need to set those since we're using traditional MCTS,
    #   but they're here for completeness
    num_considered_actions = 7,
    mcts_value_scale = 1f0,
    mcts_max_visit_init = 50,

    # AlphaZero MCTS variable
    c_puct = 2.0f0,
    alpha_dirichlet = 0.0f0,
    epsilon_dirichlet = 0.0f0,
    tau = 1.0f0,
    collapse_tau_move = 1,
)

# # play against MCTS
 #play_with_mcts(nn, mcts_kwargs, true)

# play against the neural network
#play_with_nn(nn, true)



function BinaryToBase16k(p)

    out = UInt16[]
    code = 0

    for (k, bytevalue) in enumerate(p)
        i = (k - 1) % 7
        byteValue = UInt16(reinterpret(UInt8, bytevalue))
        if i == 0
            code = byteValue << 6
        elseif i == 1
            code |= byteValue >> 2
            code += 0x5000
            push!(out, code)
            code = (byteValue & 3) << 12
        elseif i == 2
            code |= byteValue << 4

        elseif i == 3
            code |= byteValue >> 4
            code += 0x5000
            push!(out, code)
            code = (byteValue & 0xf) << 10
        elseif i == 4
            code |= byteValue << 2
        elseif i == 5
            code |= byteValue >> 6
            code += 0x5000
            push!(out, code)
            code = (byteValue & 0x3f) << 8

        elseif i == 6
            code |= byteValue
            code += 0x5000
            push!(out, code)
            code = 0
        end
    end


    if (length(p) % 7 != 0)
        code += 0x5000
        push!(out, code)
    end

    return transcode(String, out)
end

function adapt_weights(reseau, n)
    C = zeros(Float32, n, 126)
    b = [(6 - div(i - 1, 9))*9 + (i - 1) % 9+1 for i in 1:63]
    for i in 1:63
        C[:, i] .= reseau[1].weight[:, b[i]]
        C[:, i+63] .= reseau[1].weight[:, b[i]+63]
    end
    reseau[1].weight .= C
end

function weight_quant(d, index)
    QA = 127
    QB = 512
    OFF = 5000
    w = round.(Int8, QA * d[1].weight)
    ws = BinaryToBase16k(reshape(w, length(w)))
    b = round.(UInt16, (QA * d[1].bias .+ OFF)) .+ 0x5000
    bs = transcode(String, reshape(b, length(b)))

    open("/home/fabrice/C++/CodinGame/weights$index.dat", "w") do io
        write(io, "wstring w0=L\"")
        write(io, ws)
        write(io, "\";\n")
        write(io, "wstring b0 =L\"")
        write(io, bs)
        write(io, "\";\n")

        # write(io, "wstring wp=L\"")
        # write(io, wsp)
        # write(io, "\";\n")
        # write(io, "wstring bp =L\"")
        # write(io, bsp)
        # write(io, "\";\n")
    end

    for i in 2:length(d)
        if i < 2

            w = round.(Int8, (64 * d[i].weight))
            ws = BinaryToBase16k(reshape(transpose(w), length(w)))
        else

            w = round.(UInt16, (QB* d[i].weight .+ OFF)) .+ 0x5000
            ws = transcode(String, reshape(transpose(w), length(w)))
        end

        b = round.(UInt16, (QB * d[i].bias .+ OFF)) .+ 0x5000
        bs = transcode(String, reshape(b, length(b)))
        id = i - 1
        open("/home/fabrice/C++/CodinGame/weights$index.dat", "a") do io
            write(io, "wstring w$id =L\"")
            write(io, ws)
            write(io, "\";\n")

            write(io, "wstring b$id =L\"")
            write(io, bs)
            write(io, "\";\n")
        end
    end

end