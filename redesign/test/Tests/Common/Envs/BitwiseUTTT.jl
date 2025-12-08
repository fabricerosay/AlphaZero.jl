module BitwiseUTTT

using StaticArrays

using ....BatchedEnvs
using ....Util.StaticBitArrays

export BitwiseUtttEnv

const NUM_COLUMNS = 9
const NUM_ROWS = 9


const CROSS = true
const NOUGHT = false

"""
Bitboard representation:

 1 -  7  ---->   · · · · · · ·       · · · · · · ·   <---  43 - 49
 8 - 14  ---->   · · · · · · ·       · · · · · · ·   <---  50 - 56
15 - 21  ---->   · · · · · · ·       · · · · · · ·   <---  57 - 63
22 - 28  ---->   · · · · · · ·       · · · · · · ·   <---  64 - 70
29 - 35  ---->   · · · · · · ·       · · · · · · ·   <---  71 - 77
36 - 42  ---->   · · · · · · ·       · · · · · · ·   <---  78 - 84

                 NOUGHT PLAYER       CROSS PLAYER
"""
const bitboard = StaticBitArray{189, 3}

"""
A connect-four environment implemented using bitwise operations
that can be run on GPU.
"""
struct BitwiseUtttEnv
    board::bitboard
    zone::Int8
    curplayer::Bool
end

BitwiseUtttEnv() = BitwiseUtttEnv(bitboard(), CROSS)

BatchedEnvs.state_size(::Type{BitwiseUtttEnv}) = (NUM_ROWS * (NUM_COLUMNS+1) * 2,)
BatchedEnvs.num_actions(::Type{BitwiseUtttEnv}) = NUM_COLUMNS*NUM_COLUMNS

posidx(n, player) = n + (NUM_ROWS * (NUM_COLUMNS+1)) * player
posidx(x, y, player) = posidx(NUM_COLUMNS * (x - 1) + y, player)

function Base.show(io::IO, ::MIME"text/plain", env::BitwiseUtttEnv)
    string_repr = Base.string(env)
    print(io, string_repr)
end

function Base.show(io::IO, env::BitwiseUtttEnv)
    string_repr = Base.string(env)
    print(io, string_repr)
end

function Base.string(env::BitwiseUtttEnv)
    X, O = CROSS, NOUGHT
    curplayer = (env.curplayer == X) ? "X" : "O"
    s = "$curplayer to play:\n\n"
    for i in 1:NUM_ROWS
        for j in 1:NUM_COLUMNS
            s *= env.board[posidx(i, j, X)] ? "X" : (env.board[posidx(i, j, O)] ? "O" : "·")
            (j < NUM_COLUMNS) && (s *= " ")
        end
        s *= "\n"
    end
    return s
end

function BatchedEnvs.act(env::BitwiseUtttEnv, action)
    at(i, j, player) = env.board[posidx(i, j, player)]

    curr_row = NUM_ROWS
    while at(curr_row, action, env.curplayer) || at(curr_row, action, !env.curplayer)
        curr_row -= 1
    end

    board = Base.setindex(env.board, true, posidx(curr_row, action, env.curplayer))
    newenv = BitwiseUtttEnv(board, !env.curplayer)
    reward = is_win(newenv, env.curplayer) ? 1 : (is_win(newenv, !env.curplayer) ? -1 : 0)
    return newenv, (; reward, switched=true)
end

function BatchedEnvs.valid_action(env::BitwiseUtttEnv, action)
    return begin
        !env.board[posidx(1, action, env.curplayer)] &&
        !env.board[posidx(1, action, !env.curplayer)]
    end
end

function full_board(env::BitwiseUtttEnv)
    return !any(BatchedEnvs.valid_action(env, action) for action in 1:NUM_COLUMNS)
end

function is_win(env::BitwiseUtttEnv, player::Bool)
    at(i, j) = env.board[posidx(i, j, player)]
    for i in 1:NUM_ROWS, j in 1:NUM_COLUMNS
        # ToDo: Adapt this so that it works for any `TO_CONNECT`
        if at(i, j) == 1
            if i+3 <= NUM_ROWS && at(i+1, j) == at(i+2, j) == at(i+3, j) == 1
                return true
            elseif j+3 <= NUM_COLUMNS && at(i, j+1) == at(i, j+2) == at(i, j+3) == 1
                return true
            elseif (i+3 <= NUM_ROWS && j+3 <= NUM_COLUMNS) && at(i+1, j+1) == at(i+2, j+2) == at(i+3, j+3) == 1
                return true
            elseif (i >= 4 && j+3 <= NUM_COLUMNS) && at(i-1, j+1) == at(i-2, j+2) == at(i-3, j+3) == 1
                return true
            end
        end
    end
    return false
end

function BatchedEnvs.terminated(env::BitwiseUtttEnv)
    is_win(env, env.curplayer) || is_win(env, !env.curplayer) || full_board(env)
end

function BatchedEnvs.reset(::BitwiseUtttEnv,m)
    env=BitwiseUtttEnv()
    # while true
    #     m1=m%9
    #     m=div(m-m1,9)
    #     if m1==0
    #         break
    #     else
    #         env,_=BatchedEnvs.act(BitwiseUtttEnv(),m1)
    #     end
    # end
    return env
end

function get_player_board(env::BitwiseUtttEnv, player)
    return @SVector [env.board[posidx(i, player)] for i in 1:(NUM_ROWS * NUM_COLUMNS)]
end

"""
    vectorize_state(env::BitwiseUtttEnv)

Create a vectorize representation of the board.
The board is represented from the perspective of the next player to play.
It is a flatten 2x7x6 array with the following channels:
    [next player, other player]
"""
function BatchedEnvs.vectorize_state(env::BitwiseUtttEnv)
    nbrd = get_player_board(env, NOUGHT)
    cbrd = get_player_board(env, CROSS)
    order = (env.curplayer == NOUGHT) ? (@SVector [nbrd, cbrd]) : (@SVector [cbrd, nbrd])
    return Float32.(reduce(vcat, order))
end

end
