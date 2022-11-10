"""
Centered convolution between `kernel` and `input`, i.e.
`output = fftshift(ifft(fftshift( A )))`
where `A = kernel*fftshift(fft(fftshift(input)))`.

It does not matter if `input === output`.
"""
struct UniformConv{F,R}
    imfft::F
    offset::Array{Int,2}
end

function UniformConv(::Type{T}, sz::Dims{N}) #=
    =# where T<:AbstractArray{E,N} where {E<:Union{ComplexF32,ComplexF64},N}

    dummy = T(undef, sz)
    imfft = ImplicitFFT(dummy)

    offset = Array{Int,2}(undef, N, 2^N)
    set_offset!(offset)

    UniformConv{typeof(imfft),typeof(offset)}(imfft, offset)
end

const ValidArrType = AbstractArray{<:Union{ComplexF32,ComplexF64}}
function (UC::UniformConv)(output::T, kernel::A, input::T) #=
    =# where {T<:ValidArrType,A<:ValidArrType}

    input_sz = size(input)
    kernel_part_copy_sz = T == A ? Tuple(zeros(Int, ndims(output))) : input_sz

    kernel_part_copy_cpu = A(undef, kernel_part_copy_sz)
    kernel_part_copy_gpu = T(undef, kernel_part_copy_sz)
    output_part = T(undef, input_sz)
    
    # do this because it is possible that input == output
    if output === input
        input_copy = T(undef, input_sz)
    else
        input_copy = input
    end
    
    UC(output, kernel, input, input_copy, output_part, kernel_part_copy_cpu, kernel_part_copy_gpu)
end

function (UC::UniformConv)(
    output, kernel, input, input_copy, output_part, 
    kernel_part_copy_cpu, kernel_part_copy_gpu)

    copyto!(input_copy, input) 
    offset = UC.offset
    for idx in 1:size(offset,2)
        sn = offset[:,idx]
        copyto!(output_part, input_copy)
        UC.imfft(output_part, 1, false, sn...)
        convmul!(output_part, kernel, kernel_part_copy_cpu, kernel_part_copy_gpu, sn...)
        UC.imfft(output_part, -1, false, sn...)

        if idx == 1
            copyto!(output, output_part)
        else
            @inbounds output .+= output_part
        end
    end
end

function convmul!(output_part::T, kernel::A, 
    kernel_part_copy_cpu::A, kernel_part_copy_gpu::T,
    offset::Vararg{Int,D}) where {T,A,D}

    slices = Tuple((offset[d]+1):2:(2*size(output_part,d)) for d in 1:D)

    if T != A
        convmul_copy!(output_part, kernel, kernel_part_copy_cpu, kernel_part_copy_gpu, slices)
    else
        convmul!(output_part, kernel, slices)
    end
end

function convmul_copy!(output_part, kernel, kernel_part_copy_cpu, kernel_part_copy_gpu, slices)
    copyto!(kernel_part_copy_cpu, view(kernel, slices...))
    copyto!(kernel_part_copy_gpu, kernel_part_copy_cpu)
    @inbounds output_part .*= kernel_part_copy_gpu
end
convmul!(output_part, kernel, slices) = @inbounds output_part .*= view(kernel, slices...)

