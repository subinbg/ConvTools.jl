"""
In-place, centered, padded fast Fourier transform.  

If `full=false`, some exponentials are omitted 
to reduce computations in the center-oriented convolution.
If `full=true`, `ImplicitFFT` calculates the actual center-oriented
Fourier coeffcients.
"""
struct ImplicitFFT{N,T,FT,IFT}
    expf::Dict{String,T}
    F::FT
    F⁻::IFT

    function ImplicitFFT(dummy::T) where T<:AbstractArray{<:Union{ComplexF32,ComplexF64},D} where D
        0 < D < 4 ? nothing : error("ImplicitFFT: unsupported dimension, $D")
        all(size(dummy) .% 2  .== 0) ? nothing : #=
        =# error("ImplicitFFT: matrices should have even pixels in all dimensions")

        F  = plan_fft!(dummy)
        F⁻ = plan_ifft!(dummy)
        
        sz = size(dummy)
        grid = []
        for dim in 1:D
            shp = circshift(cat([sz[dim]], ones(Int, D-1), dims=1), dim-1)
            push!(grid, reshape(0:(sz[dim]-1), shp...))
        end

        πi = π * im
        expf = Dict{String, T}()
        for dim = 1:D
            for offset = 0:1
                expf[string(dim,offset,"fw1")]  = T((-1).^grid[dim] .* exp.(-πi*offset*grid[dim]/sz[dim]))
                expf[string(dim,offset,"fw2")]  = T((1im).^(2*grid[dim] .+ offset) .* (-1im)^sz[dim])
                expf[string(dim,offset,"iv1")]  = T((-1).^grid[dim] * (-1im)^offset * (1im)^sz[dim])
                expf[string(dim,offset,"iv2")]  = T((-1).^grid[dim] .* exp.(πi*offset*grid[dim]/sz[dim]) / 2.0)
            end
        end

        new{D,T,typeof(F),typeof(F⁻)}(expf, F, F⁻)
    end
end


function forward(imF::ImplicitFFT{N,T}, input::T, full::Bool, offset::Vararg{Int,N}) #=
    =# where {T<:AbstractArray{<:Complex},N}

    for dim = 1:N
        @inbounds input .*= imF.expf[string(dim,offset[dim],"fw1")]
    end
    imF.F*input
    if full
        for dim = 1:N
            @inbounds input .*= imF.expf[string(dim,offset[dim],"fw2")]
        end
    end
end

function adjoint(imF::ImplicitFFT{N,T}, input::T, full::Bool, offset::Vararg{Int,N}) #=
    =# where {T<:AbstractArray{<:Complex},N}

    if full
        for dim = 1:N
            @inbounds input .*= imF.expf[string(dim,offset[dim],"iv1")]
        end
    end
    imF.F⁻*input
    for dim = 1:N
        @inbounds input .*= imF.expf[string(dim,offset[dim],"iv2")]
    end
end

function (imF::ImplicitFFT{N,T})(input::T, direction::Int, full::Bool, offset::Vararg{Int,N}) #=
    =# where {T<:AbstractArray{<:Complex},N}
    if direction > 0
        forward(imF, input, full, offset...)
    else
        adjoint(imF, input, full, offset...)
    end
end
