module Util

using Reexport

include("StaticBitArrays.jl")
@reexport using .StaticBitArrays

include("Devices.jl")
@reexport using .Devices: Devices, CPU, GPU

include("Bitboard.jl")
@reexport using .Bitboard
end
