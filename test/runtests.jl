using CUDA
using FFTW
using Test
using Random
using ConvTools

CUDA.allowscalar(false)
FFTW.set_num_threads(Threads.nthreads())
if (Threads.nthreads() == 1)
    @warn "(# of threads) is 1. Turn on the multithreading by e.g. `julia -t auto ...`"
end

function test_fourier_transform(imF::ImplicitFFT{N,T}, input::T, bench::Bool) where {N,T}

    randn!(input)
    input_copy1 = T(undef, size(input))
    input_copy2 = T(undef, size(input))
    input_copy3 = T(undef, size(input))
    input_pad = T(undef, size(input).*2)

    do_fft = plan_fft(input_pad)
    forward = x -> fftshift(do_fft*fftshift(x))

    fill!(input_pad, 0)
    fill!(input_copy3, 0)
    copyto!(view(input_pad, [(n÷2+1):(3*n÷2) for n in size(input)]...), input)

    Ff = forward(input_pad)

    offset = Array{Int,2}(undef, N, 2^N)
    ConvTools.set_offset!(offset)
    checks = Array{Bool,1}(undef, 2^N+1)

    for idx in 1:size(offset,2)
        copyto!(input_copy1, input)
        imF(input_copy1, 1, true, offset[:,idx]...)
        copyto!(input_copy2, view(Ff, [(offset[d,idx]+1):2:(2*size(input,d)) for d in 1:N]...))

        checks[idx] = isapprox(input_copy1, input_copy2)
        
        imF(input_copy2, -1, true, offset[:,idx]...)
        copyto!(input_copy1, input_copy2)
        
        input_copy3 .+= input_copy1
    end
    checks[end] = isapprox(input, input_copy3)

    # if bench
    #     bench = @benchmark begin
    #         Ff = $forward($input_pad)
    #     end
    #     t_normal = BenchmarkTools.prettytime(time(median(bench)))
        
    #     bench = @benchmark begin
    #         for idx in 1:size($offset,2)
    #             $imF($input_copy1, 1, true, $offset[:,idx]...)
    #         end
    #     end
    #     t_implicit = BenchmarkTools.prettytime(time(median(bench)))
        
    #     return t_normal, t_implicit
    # end

    return checks
end


function test_convolution(UC::UniformConv, input::T, cpukernel::Bool) where T

    randn!(input)
    input_copy1 = T(undef, size(input))
    input_copy2 = T(undef, size(input))
    input_pad = T(undef, size(input).*2)
    kernel = cpukernel ? Array{eltype(T),ndims(T)}(undef, size(input_pad)) : T(undef, size(input_pad))
    randn!(kernel)

    do_fft = plan_fft(input_pad)
    do_ifft = plan_ifft(input_pad)
    forward = x -> fftshift(do_fft*fftshift(x))
    adjoint = x -> fftshift(do_ifft*fftshift(x))

    fill!(input_pad, 0)
    copyto!(view(input_pad, [(n÷2+1):(3*n÷2) for n in size(input)]...), input)

    Ff = forward(input_pad)
    Gf = adjoint(Ff .* T(kernel))
    copyto!(input_copy1, view(Gf, [(n÷2+1):(3*n÷2) for n in size(input)]...))

    copyto!(input_copy2, input)

    UC(input_copy2, kernel, input_copy2)

    return isapprox(input_copy1, input_copy2)
end


@testset "ConvTools.jl" begin
    arrtypes = CUDA.functional() ? (Array, CuArray) : (Array,) 
    for arrtype in arrtypes
        for fp in (ComplexF32, ComplexF64)
            for dim in (1,2,3)
                @testset "($arrtype, $fp, $dim)" begin
                    input = arrtype{fp,dim}(undef, [2*rand(50:80) for n in 1:dim]...)
                    imF = ImplicitFFT(input)
                    checks = test_fourier_transform(imF, input, false)
                    @test all(checks)
                end
            end
        end
    end

    for arrtype in arrtypes
        for fp in (ComplexF32, ComplexF64)
            for dim in (1,2,3)
                for cpukernel in (true, false)
                    @testset "($arrtype, $fp, $dim, cpukernel=$cpukernel)" begin
                        input = arrtype{fp,dim}(undef, [2*rand(50:80) for n in 1:dim]...)
                        UC = UniformConv(typeof(input), size(input))
                        check = test_convolution(UC, input, cpukernel)
                        @test all(check)
                    end
                end
            end
        end
    end
end
