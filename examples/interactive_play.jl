using RLZero.BatchedMcts
using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Tests.Common.BitwiseHex
using RLZero.TrainUtilities: init_mcts_config
using RLZero.Util.Devices
using RLZero
using Flux
using JLD2

const MCTS = BatchedMcts


const N = 7
const A_CODE = Int('A') # ASCII code for 'A'

# --- Indexing Function (copied from BitwiseHexEnv) ---
# Converts 1-based (row, col) to the flat 1-based index (1 to N*N)
rc_to_n(r, c) = (r - 1) * N + c

# Converts flat 1-based index (1 to N*N) to (row, col)
n_to_rc(n) = ((n - 1) ÷ N + 1, (n - 1) % N + 1)


## --- 1. Algebraic Notation to Flat Index ---

"""
    alg_to_flat(alg::String)::Union{Int, Nothing}

Converts an algebraic Hex move string (e.g., "A1", "G7") to the flat 1-based
integer index (1 to 49) used by the BitwiseHexEnv. Returns `nothing` if invalid.

# Examples
- alg_to_flat("A1") == 1
- alg_to_flat("G7") == 49
- alg_to_flat("D4") == 25
"""
function alg_to_flat(alg::String)::Union{Int, Nothing}
    alg = uppercase(strip(alg))

    # Basic format check (e.g., "A1" or "G7")
    if !(2 <= length(alg) <= 3)
        return nothing
    end

    # 1. Parse Column (Letter A-G)
    col_char = alg[1]
    col_int = Int(col_char) - A_CODE + 1
    if !(1 <= col_int <= N)
        return nothing
    end

    # 2. Parse Row (Number 1-7)
    row_str = alg[2:end]
    try
        row_int = parse(Int, row_str)
        if !(1 <= row_int <= N)
            return nothing
        end
        
        # 3. Calculate Flat Index
        # Note: Hex notation (Col, Row) maps to our internal (Row, Col)
        # rc_to_n(row, col)
        return rc_to_n(row_int, col_int)
    catch
        return nothing # Failed to parse row number
    end
end


## --- 2. Flat Index to Algebraic Notation ---

"""
    flat_to_alg(n::Int)::Union{String, Nothing}

Converts a flat 1-based integer index (1 to 49) to the algebraic Hex move string
(e.g., "A1", "G7"). Returns `nothing` if the index is out of bounds.

# Examples
- flat_to_alg(1) == "A1"
- flat_to_alg(49) == "G7"
- flat_to_alg(25) == "D4"
"""
function flat_to_alg(n)::Union{String, Nothing}
    if !(1 <= n <= N*N)
        return nothing
    end

    # 1. Get Row and Column indices
    row_int, col_int = n_to_rc(n)

    # 2. Convert Column index (1-7) to Letter (A-G)
    col_char = Char(A_CODE + col_int - 1)

    # 3. Combine
    return string(col_char, row_int)
end
# set these constants to your preference
const DEVICE = CPU()
const MODEL_PATH = "examples/models/hex-checkpoints/model_05000.jld2"
const nn_config = SimpleResNetHP(
    width=512,
    depth_common=6,
    depth_vhead=1,
    depth_phead=1
)


function load_nn()
    state_dim = BatchedEnvs.state_size(BitwiseHexEnv)
    action_dim = BatchedEnvs.num_actions(BitwiseHexEnv)
    nn = SimpleResNet(state_dim..., action_dim, nn_config)

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
    println("AlphaZero chooses action: $(flat_to_alg(action))\n\n", " value :", RLZero.BatchedMcts.value(tree|>cpu,1,1))
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
    println("Raw Neural Network chooses action: $(flat_to_alg(action))\n\n")
    return action
end

function _parse_player_action(env)
    na = BatchedEnvs.num_actions(typeof(env))
    valid_actions = findall(!iszero, [BatchedEnvs.valid_action(env, a) for a in 1:na])
    print("Input action: ")
    alg_action =readline()
    println("alg_action: $alg_action")
    action=alg_to_flat(alg_action)
    println("action $action")
    while isnothing(action) || action ∉ valid_actions
        print("Invalid action. Please choose one of $valid_actions. Input action: ")
        alg_action =readline()
        action=alg_to_flat(alg_action)
    end
    println("User chooses action: $action\n\n")
    return action
end

"""Play against an MCTS player using the provided neural network."""

posidx(n, player) = n + (63) * player
posidx(x, y, player) = posidx(9 * (x - 1) + y, player)
# function get_action_id(env::BitwiseConnectFourEnv, action)
#     at(i, j, player) = env.board[posidx(i, j, player)]

#     curr_row = 7
#     while at(curr_row, action, env.curplayer) || at(curr_row, action, !env.curplayer)
#         curr_row -= 1
#     end
  
#     return  posidx(curr_row, action, env.curplayer)%63
   
# end
function play_with_mcts(nn, mcts_kwargs, az_goes_first = true)
    mcts_config = init_mcts_config(DEVICE, nn, mcts_kwargs)
   
    env = BitwiseHexEnv()
    #env,_=BatchedEnvs.act(env, 8)
    #println(env)
    
    az_plays = az_goes_first
    done, info = false, nothing
    while !done
        
        t=@elapsed action =az_plays ? _mcts_action(env, mcts_config) : _parse_player_action(env)
        println("time: $t")
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
    num_simulations = 600,

    # Gumbel MCTS variables -- No need to set those since we're using traditional MCTS,
    #   but they're here for completeness
    num_considered_actions = 9,
    mcts_value_scale = 1f0,
    mcts_max_visit_init = 50,

    # AlphaZero MCTS variable
    c_puct = 1.0f0,
    alpha_dirichlet = 0.0f0,
    epsilon_dirichlet = 0.0f0,
    tau = 1.0f0,
    collapse_tau_move = 1,
)

# # play against MCTS
 play_with_mcts(nn, mcts_kwargs, true)

# play against the neural network
#play_with_nn(nn, true)


