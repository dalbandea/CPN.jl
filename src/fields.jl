
@doc raw"""
    struct CPworkspace{T}

Allocates all the necessary fields for a HMC simulation of a CPN model:

- `PRC`: precision; must be `Float16`, `Float32` or `Float64`.
- `x`: real `2N` vector field representation of the complex `N` vector field ``z``.
- `phi`: angles of the gauge field ``\lambda_{n,\mu} = \exp (i\phi_{n,\mu})``.
- `J_n`
- `Lambda`
- `frc_x`
- `frc_phi`
- `I_N`: identity matrix of size `NxN`
- `P_n`: projector to the tangent space of `x` at the point `n`.
- `Gamma`
- `Lambda_block`
"""
struct CPworkspace{T, N1, N2}
    PRC::Type{T}
    x::Array{T, N1}
    phi::Array{T, N1}
    x_cp::Array{T, N1}
    J_n::Array{T,N1}
    Lambda::Array{T,N2}
    frc_x::Array{T, N1}
    frc_phi::Array{T, N1}
    I_N
    P_n
    Gamma
    Lambda_block
    function CPworkspace(::Type{T}, lp::LattParm) where {T <: AbstractFloat}
        x = Array{T, 3}(undef, 2 * lp.N, lp.iL...)
        phi = Array{T, 3}(undef, 2, lp.iL...)
        x_cp = similar(x)
        J_n = similar(x)
        Lambda = Array{T, 5}(undef, 2*lp.N, 2*lp.N, 2, lp.iL...)
        frc_x = similar(x)
        frc_phi = similar(phi)
        I_N = Matrix(I, lp.N, lp.N)
        P_n = Array{T, 2}(undef, 2*lp.N, 2*lp.N)
        Gamma = kron(I_N, [0 -1; 1 0])
        Lambda_block = Array{T,5}(undef, 2, 2, size(phi)...)
        return new{T, 3, 5}(T, x, phi, x_cp, J_n, Lambda, frc_x, frc_phi, I_N, P_n, Gamma, Lambda_block)
    end
end

@doc raw"""
    project_to_Sn!(x, lp::LattParm)

Normalizes the vector field `x` at every lattice point.
"""
function project_to_Sn!(x, lp::LattParm)
    for j in 1:lp.iL[1]
        for i in 1:lp.iL[2]
            @views LinearAlgebra.normalize!(x[:,i,j])
        end
    end
    return nothing
end

function randomize_x!(x, lp::LattParm)
    x .= Random.rand(eltype(x), size(x)...) .* 2 .- 1
    project_to_Sn!(x, lp)
    return nothing
end


@doc raw"""
    function randomize!(cpws, lp)

Randomizes:

- `cpws.phi` uniformly between `[0, 2pi]`.
- `cpws.x` uniformly between `[-1, 1]` and then normalize to 1.

Also sets `cpws.J_n` and `cpws.Lambda` to zero.
"""
function randomize!(cpws::CPworkspace, lp::LattParm)
    cpws.phi .= Random.rand(cpws.PRC, size(cpws.phi)...) * 2 * pi
    randomize_x!(cpws.x, lp)
    cpws.Lambda .= zero(cpws.PRC)
    cpws.J_n .= zero(cpws.PRC)
    return nothing
end


function fill_Lambda!(cpws::CPworkspace, lp::LattParm)
    fill_Lambda!(cpws.Lambda, cpws.Lambda_block, cpws.phi, cpws.I_N, cpws, lp)
end

function fill_Lambda!(Lambda, Lambda_block, phi, I_N, cpws::CPworkspace, lp::LattParm)
    # Lambda_block .= cat( cos.(phi), sin.(phi), -sin.(phi), cos.(phi), dims=1 ) |> x -> reshape(x, (2, 2, size(phi)...)) # beware column major! stores first in columns!
    Lambda_block .= cat( cos.(phi), sin.(phi), -sin.(phi), cos.(phi), dims=4) |> x -> permutedims(x, (4,1,2,3)) |> x -> reshape( x, (2,2,size(phi)...))
    for j in 1:lp.iL[1]
        for i in 1:lp.iL[2]
            for mu in 1:2
                @views Lambda[:,:,mu,i,j] .= kron(I_N, Lambda_block[:,:,mu,i,j])
            end
        end
    end
    return nothing
end

function fill_Jn!(cpws::CPworkspace, lp::LattParm)
    fill_Jn!(cpws.J_n, cpws.x, cpws.Lambda, cpws, lp)
end

function fill_Jn!(J_n, x, Lambda, cpws::CPworkspace, lp::LattParm)
    J_n .= zero(eltype(J_n))
    for j in 1:lp.iL[1]
        for i in 1:lp.iL[2]
            iu = ((i+1) - 1) % lp.iL[1] + 1
            ju = ((j+1) - 1) % lp.iL[2] + 1
            id = ((i-1) - 1 + lp.iL[1]) % lp.iL[1] + 1
            jd = ((j-1) - 1 + lp.iL[2]) % lp.iL[2] + 1
            @views J_n[:,i,j] .= J_n[:,i,j] .+ transpose(Lambda[:,:,1,i,j]) * x[:,iu,j]
            @views J_n[:,i,j] .= J_n[:,i,j] .+ transpose(Lambda[:,:,2,i,j]) * x[:,i,ju]
            @views J_n[:,i,j] .= J_n[:,i,j] .+ Lambda[:,:,1,id,j] * x[:,id,j]
            @views J_n[:,i,j] .= J_n[:,i,j] .+ Lambda[:,:,2,i,jd] * x[:,i,jd]
            # Lines below would avoid memory allocation but execution time is
            # higher
            # @views mult_add!(J_n[:,i,j], transpose(Lambda[:,:,1,i,j]), x[:,iu,j], lp)
            # @views mult_add!(J_n[:,i,j], transpose(Lambda[:,:,2,i,j]), x[:,i,ju], lp)
            # @views mult_add!(J_n[:,i,j], Lambda[:,:,1,i,jd], x[:,id,j], lp)
            # @views mult_add!(J_n[:,i,j], Lambda[:,:,2,i,jd], x[:,i,jd], lp)
        end
    end
    return nothing
end

function mult_add!(vo, A, vi, lp::LattParm)
    for j in 1:2*lp.N
        for k in 1:2*lp.N
            @inbounds vo[j] += A[j,k] * vi[k]
        end
    end
end

"""
    sync_fields!(cpws::CPworkspace, lp::LattParm)

Updates `cpws.Lambda` and `cpws.J_n` with current `cpws.x` and `cpws.phi`.
"""
function sync_fields!(cpws::CPworkspace, lp::LattParm)
    fill_Lambda!(cpws, lp)
    fill_Jn!(cpws, lp)
    return nothing
end
