struct NonUniformConv{C<:Complex,
    nu<:Tuple, P1<:Pnufft.Plan, P2<:Pnufft.Plan} <: AbstractConv

    nufftplan1::P1
    nufftplan2::P2
    nupts::nu
    NF::Int

    function NonUniformConv(nupts::NTuple{D,T}, N::Dims{D}; NF::Int=-1, tol=1e-6) #=
        =# where {T<:AbstractArray{R,1}, D} where R<:Union{Float32,Float64}

        Nnupts = length(nupts[1])
        for i=1:D
            @assert Nnupts==length(nupts[i]) "Nupts mismatch in the $(i)-th dimension."
        end

        plan1 = makeplan(1,  1, R, N, Nnupts, tol=tol)
        plan2 = makeplan(2, -1, R, N, Nnupts, tol=tol)
        
        nupts_cuda = CuArray.(nupts)

        # NF = normalization factor
        # In general, NF = (pad_factor)^D * N1 * ... * ND
        if NF < 0 # automatic setting
            NF = *(N...) * (2^D)  # length(nupts[1]) #
        end
        
        typ = [complex(R),typeof(nupts_cuda),typeof(plan1),typeof(plan2)]
        new{typ...}(plan1, plan2, nupts_cuda, NF)
    end
end


# It does not matter if input === output.
function (NC::NonUniformConv{C})(output::T, kernel::A, input::T) #=
    =# where {T<:CuArray{C}, A<:CuArray{C,1}} where C<:Complex

    Finput = A(undef, size(kernel))
    NC(output, kernel, input, NC.nupts, Finput)
end


function (NC::NonUniformConv{C})(output::T, kernel::A, input::T, nupts::nu, Finput::A) #=
    =# where {T<:CuArray{C}, A<:CuArray{C,1}, nu} where C<:Complex

    execute!(NC.nufftplan2, Finput, input, nupts...)
    Finput .*= kernel
    execute!(NC.nufftplan1, Finput, output, nupts...)
    output ./= NC.NF 
end