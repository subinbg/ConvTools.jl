module ConvTools

export ImplicitFFT
export AbstractConv, UniformConv
export NonUniformConv

using FFTW
using CUDA
# CUDA.allowscalar(false)

using Adroit
using Pnufft

abstract type AbstractConv end

include("utils.jl")
include("ImplicitFFT.jl")
include("UniformConv.jl")
include("NonUniformConv.jl")

end