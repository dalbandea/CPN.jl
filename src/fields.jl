
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
        x = Array{T, 3}(undef, lp.iL..., 2 * lp.N)
        phi = Array{T, 3}(undef, lp.iL..., 2)
        x_cp = similar(x)
        J_n = similar(x)
        Lambda = Array{T, 5}(undef, lp.iL..., 2, 2*lp.N, 2*lp.N)
        frc_x = similar(x)
        frc_phi = similar(phi)
        I_N = Matrix(I, lp.N, lp.N)
        P_n = Array{T, 2}(undef, 2*lp.N, 2*lp.N)
        Gamma = kron(I_N, [0 -1; 1 0])
        Lambda_block = Array{T,5}(undef, size(phi)..., 2, 2)
        return new{T, 3, 5}(T, x, phi, x_cp, J_n, Lambda, frc_x, frc_phi, I_N, P_n, Gamma, Lambda_block)
    end
end

@doc raw"""
    project_to_Sn!(x, lp::LattParm)

Normalizes the vector field `x` at every lattice point.
"""
function project_to_Sn!(x, lp::LattParm)
    for i in 1:lp.iL[1]
        for j in 1:lp.iL[2]
            @views x[i,j,:] .= x[i,j,:] / sqrt(dot(x[i,j,:], x[i,j,:]))
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
    Lambda_block .= cat( cos.(phi), sin.(phi), -sin.(phi), cos.(phi), dims=4 ) |> x -> reshape(x, (size(phi)..., 2, 2)) # beware column major! stores first in columns!
    for i in 1:lp.iL[1]
        for j in 1:lp.iL[2]
            for mu in 1:2
                @views Lambda[i,j,mu,:,:] .= kron(I_N, Lambda_block[i,j,mu,:,:])
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
            @views J_n[i,j,:] .= J_n[i,j,:] .+ transpose(Lambda[i,j,1,:,:]) * x[iu,j,:]
            @views J_n[i,j,:] .= J_n[i,j,:] .+ transpose(Lambda[i,j,2,:,:]) * x[i,ju,:]
            @views J_n[i,j,:] .= J_n[i,j,:] .+ Lambda[id,j,1,:,:] * x[id,j,:]
            @views J_n[i,j,:] .= J_n[i,j,:] .+ Lambda[i,jd,2,:,:] * x[i,jd,:]
        end
    end
    return nothing
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