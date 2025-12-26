module BitwiseHex

using StaticArrays

using ....BatchedEnvs
using ....Util.Bitboard

export BitwiseHexEnv

const N=7
const NN=N*N
const VectorizedState=(N+1)*(N+1)
const maxActions=NN
const maxLengthGame=NN




struct BitwiseHexEnv
	bplayer::bitboard{2}
	bopponent::bitboard{2}
	player::Int8
	lp::Int8
end

BatchedEnvs.state_size(::Type{BitwiseHexEnv}) = (2*VectorizedState,)

BatchedEnvs.num_actions(::Type{BitwiseHexEnv}) = NN

const empty=bitboard{2}(N+1,N+1)

function init()
	startx=empty
	starto=empty
	for i in 3:N+1
		startx=Bitboard.setindex(startx,true,i,1)
		starto=Bitboard.setindex(starto,true,1,i)
	end
	startx,starto
end



const startx,starto=init()

BitwiseHexEnv()=BitwiseHexEnv(startx,starto,1,NN)

# function Position(n)
# 	pos=Position()
# 	bplayer=pos.bplayer
# 	bopponent=pos.bopponent
# 	for x in 0:N-1
# 		for y in n+1:N
# 			newcol=(N+1)*(x+1)+y+1
# 			bopponent=Bitboard.setindex(bopponent,true,newcol)
# 			newcol=(N+1)*(y)+x+1
# 			bplayer=Bitboard.setindex(bplayer,true,newcol)
# 		end
# 	end
# 	return Position(bplayer,bopponent,pos.player,n*n)
# end


"""
An action is an integer from 1 to NN, representing the flat index
of the cell to be played.
"""
function pov_action(action,player)
	if player==1
		return action
	else
		x=div(action-1,N)
		y=action-1-N*x
		return N*y+x+1
	end
end

function big_pov_action(action,player)
	if player==1
		return action
	else
		x=div(action-1,N+1)
		y=action-1-(N+1)*x
		return (N+1)*y+x+1
	end
end

function BatchedEnvs.act(env::BitwiseHexEnv, action)
	pov_act=pov_action(action,env.player)
	x=div(pov_act-1,N)
	y=pov_act-N*x
	newcol=(N+1)*(x+1)+y+1
	bplayer=Bitboard.setindex(env.bplayer,true,newcol)
	newenv= BitwiseHexEnv(env.bopponent,bplayer,-env.player,env.lp-Int8(1))
    reward = is_win(newenv) ? 1 : 0
    return newenv, (; reward, switched=true)
end

"""
A move is valid if the cell is currently empty.
The action is the flat index n (1 to N*N).
"""
function BatchedEnvs.valid_action(env::BitwiseHexEnv, action::Int)
	pov_act=pov_action(action,env.player)
	x=div(pov_act-1,N)
	y=pov_act-N*x
	newcol=(N+1)*(x+1)+y+1
	return ~env.bplayer[newcol] & ~env.bopponent[newcol]
end

function is_win(env::BitwiseHexEnv)
    a=env.bopponent
	for j in 1:2*N-2
		b=Bitboard.up(a)
		c=Bitboard.right(b)
		a=Bitboard.down((a&(b|c))|(b&c))
		if env.player==1
        	for k in 3+j:N+1
            	a=Bitboard.setindex(a,true,1,k)
        	end
		end
	end
	return a[N+1,N+1]
end

function BatchedEnvs.terminated(env::BitwiseHexEnv)
   return is_win(env)
end

function BatchedEnvs.reset(::BitwiseHexEnv, m::Int)
    env = BitwiseHexEnv()
    env, _ = BatchedEnvs.act(env, m)
    return env
end


"""
    vectorize_state(env::BitwiseHexEnv)

Create a vectorize representation of the board.
The board is represented from the perspective of the next player to play.
It is a flatten 2xN*N array with the following channels:
    [next player, other player]
"""


function BatchedEnvs.vectorize_state(env::BitwiseHexEnv)
    me=@SVector [env.bplayer[big_pov_action(n,env.player)] for n in 1:VectorizedState]
    you=@SVector [env.bopponent[big_pov_action(n,env.player)] for n in 1:VectorizedState]
    return Float32.(reduce(vcat, @SVector [me,you]))
end

function BatchedEnvs.masks(env::BitwiseHexEnv)
    return @SVector [valid_action(env,n) for n in 1:maxActions]
end

end