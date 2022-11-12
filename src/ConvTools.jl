module ConvTools

export ImplicitFFT, UniformConv, AbstractConv

using FFTW
using CUDA

abstract type AbstractConv end

include("utils.jl")
include("ImplicitFFT.jl")
include("UniformConv.jl")

end