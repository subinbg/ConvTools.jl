module ConvTools

export ImplicitFFT, UniformConv

using FFTW
using CUDA

include("utils.jl")
include("ImplicitFFT.jl")
include("UniformConv.jl")

end