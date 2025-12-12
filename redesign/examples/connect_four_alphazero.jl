using RLZero
using RLZero.BatchedEnvs
using RLZero.Network
using RLZero.Train
using RLZero.TrainUtilities: TrainConfig, get_train_timestamps, print_execution_times
using RLZero.Tests.Common.BitwiseConnectFour
using RLZero.Tests.Common.BitwiseConnectFourEvalFns
using RLZero.Util.Devices

using JLD2
using Flux
using Random

const SAVEDIR = "examples/models/connect-four-checkpoints"
const PLOTSDIR = "examples/plots/connect-four"

state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)

# large architecture
const MODEL_PATH = "examples/models/connect-four-checkpoints/model_020900.jld2"
neural_net_hyperparams = SimpleResNetHP(
    width=512,
    depth_common=6,
    depth_vhead=1,
    depth_phead=1
)
nn_cpu = SimpleResNet(state_dim..., action_dim, neural_net_hyperparams)


function load_nn()
    state_dim = BatchedEnvs.state_size(BitwiseConnectFourEnv)
    action_dim = BatchedEnvs.num_actions(BitwiseConnectFourEnv)
    nn = SimpleResNet(state_dim..., action_dim, neural_net_hyperparams)

    model_state = JLD2.load(MODEL_PATH, "model_state");
    Flux.loadmodel!(nn, model_state);
    nn = Flux.cpu(nn) 
    return nn
end

# global lists used to retrieve data from the benchmarks
metrics = Dict(
    "az" => Dict("random" => [], "minimax" => [], "benchmark" => [], "pons" => []),
    "nn" => Dict("random" => [], "minimax" => [], "benchmark" => [], "pons" => [])
)


"""Returns a list with all the evaluation functions to be called during evaluation sessions.
    Most of these functions are closures, which are being computed by the parent functions
    defined in: Tests/Common/Evaluation/EvaluationFunctions/BitwiseConnectFourEvalFns.jl.
    The evaluation functions for Pascal Pons' benchmark are not used, as they should be
    run separately. For those, another function that returns them is defined below."""
function get_eval_fns()

    # This config object will be used by `TrainUtilities.init_mcts_config()`
    #   (in TrainUtilities.jl) to create a `MctsConfig` object for the evaluation functions.
    #   It has to define all the specific-MCTS hyperparameters, even those that won't be
    #   used during evaluation. In this case, only `use_gumbel_mcts`, `num_simulations`,
    #   `num_considered_actions`, `mcts_value_scale` and `mcts_max_visit_init` will be used.
    eval_config = (;
        use_gumbel_mcts=false,

        # num_simulations = 32,
        # num_considered_actions::Int = 4,
        # mcts_value_scale::Float32 = 0.1f0,
        # mcts_max_visit_init::Int = 50,

        num_simulations=600,
        c_puct=2.0f0,
        alpha_dirichlet=0.3f0,
        epsilon_dirichlet=0.25f0,
        tau=0.3f0,
        collapse_tau_move=30
    )

    az_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 5,
        "config" => deepcopy(eval_config)
    )
    nn_vs_random_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_bilateral_rounds" => 10
    )
    az_vs_minimax_kwargs = Dict(
        "stochastic_minimax_depth" => 5,
        "stochastic_minimax_rng" => Random.MersenneTwister(42),
        "num_bilateral_rounds" => 5,
        "config" => deepcopy(eval_config)
    )
    nn_vs_minimax_kwargs = Dict(
        "stochastic_minimax_depth" => 5,
        "stochastic_minimax_rng" => Random.MersenneTwister(42),
        "num_bilateral_rounds" => 25
    )
    benchmark_fns_kwargs = Dict(
        "rng" => Random.MersenneTwister(13),
        "num_states" => 10_000,
        "device" => GPU(),
        "config" => deepcopy(eval_config)
    )
    pos_benchmark_fns_kwargs = Dict(
        "device" => GPU(),
        "config" => deepcopy(eval_config)
    )

  #alphazero_vs_random_eval_fn = get_alphazero_vs_random_eval_fn(
   #    az_vs_random_kwargs,
    #    metrics
  # )
    nn_vs_random_eval_fn = get_nn_vs_random_eval_fn(nn_vs_random_kwargs, metrics)
    alphazero_vs_minimax_eval_fn = get_alphazero_vs_minimax_eval_fn(
    az_vs_minimax_kwargs,
    metrics
    )
    nn_vs_minimax_eval_fn = get_nn_vs_minimax_eval_fn(nn_vs_minimax_kwargs, metrics)
    fns = get_connect_four_benchmark_fns(benchmark_fns_kwargs, metrics)
    mcts_benchmark_fn, nn_benchmark_fn = fns
    fns = get_pons_benchmark_fns(pos_benchmark_fns_kwargs, metrics)
    az_pons_benchmark_fn, nn_pons_benchmark_fn = fns

    return [
        #alphazero_vs_random_eval_fn,
        #nn_vs_random_eval_fn,
        #alphazero_vs_minimax_eval_fn,
        #nn_vs_minimax_eval_fn,
        mcts_benchmark_fn,
        nn_benchmark_fn,
        az_pons_benchmark_fn,
        nn_pons_benchmark_fn
    ]
end


"""Returns a `TrainConfig` object. All the hyperparameters can be set here."""
function create_config()

    # environment variables
    EnvCls = BitwiseConnectFourEnv
    env_kwargs = Dict()
    num_envs = 30_000

    # common MCTS variables
    use_gumbel_mcts = true
    num_simulations = 4
    num_considered_actions::Int = 2
    mcts_value_scale::Float32 =1.0f0
    mcts_max_visit_init::Int = 50

    # Gumbel MCTS variables
    # ...we can omit these since we're using Traditional Alphazero MCTS

    # AlphaZero MCTS variables
    c_puct = 1.7f0
    alpha_dirichlet = 0.3f0
    epsilon_dirichlet = 0.25f0
    tau = 0.3f0
    collapse_tau_move = 35

    # NN Training variables
    replay_buffer_size = num_envs * 200
    min_train_samples = 1_000
    train_freq = num_envs * 50
    adam_learning_rate = 1e-3
    weight_decay = 1e-6
    batch_size = 4096
    train_epochs = 2

    # Logging variables
    train_logfile = "train.log"
    eval_logfile = "eval.log"
    tb_logdir = "tb_logs"  # run `tensorboard --logdir tb_logs --port 6006` on a terminal

    # NN saving dir
    nn_save_dir = SAVEDIR

    # Evaluation variables
    evaluation_fns = get_eval_fns()
    eval_freq = num_envs * 1000

    # Total train steps
    num_steps = num_envs * 5000

    return TrainConfig(;
        EnvCls=EnvCls,
        env_kwargs=env_kwargs,
        num_envs=num_envs,

        use_gumbel_mcts=use_gumbel_mcts,
        num_considered_actions=num_considered_actions,
        mcts_value_scale=mcts_value_scale,
        mcts_max_visit_init=mcts_max_visit_init,
        num_simulations=num_simulations,

        c_puct=c_puct,
        alpha_dirichlet=alpha_dirichlet,
        epsilon_dirichlet=epsilon_dirichlet,
        tau=tau,
        collapse_tau_move=collapse_tau_move,

        replay_buffer_size=replay_buffer_size,
        min_train_samples=min_train_samples,
        train_freq=train_freq,
        adam_lr=adam_learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        train_epochs=train_epochs,

        train_logfile=train_logfile,
        eval_logfile=eval_logfile,
        tb_logdir=tb_logdir,

        nn_save_dir=nn_save_dir,

        eval_fns=evaluation_fns,
        eval_freq=eval_freq,

        num_steps=num_steps
    )
end

# empty the save/plot directories
#run(`rm -rf $(SAVEDIR)`)
run(`rm -rf $(PLOTSDIR)`)

# choose the device to train AlphaZero on (`CPU()` or `GPU()`)
 device = GPU()

# # get the NN in the specified device
 nn = (device == CPU()) ? nn_cpu : Flux.gpu(nn_cpu)
 #nnue=(device == CPU()) ? nnue_cpu : Flux.gpu(nnue_cpu)
# cnn = (device == CPU()) ? cnn_cpu : Flux.gpu(cnn_cpu)
# # create the config object
 config = create_config()

# # train!
 nn, execution_times = selfplay!(config, device, nn)

# # get train timestamps
 timestamps = get_train_timestamps(execution_times, config)

# # print some statistics
 print_execution_times(execution_times)

# # plot the metrics
 plot_metrics(PLOTSDIR, timestamps, metrics)
