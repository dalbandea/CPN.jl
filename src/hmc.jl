
function HMC!(cpws::CPworkspace, epsilon, ns, lp::LattParm)
    x_cp = copy(cpws.x)
    phi_cp = copy(cpws.phi)

    # Create Lambda and J_n
    sync_fields!(cpws, lp)

    # Create momenta for gauge variables
    mom_phi = Random.randn(size(cpws.phi))

    # Create and project x momenta to tangent space
    mom_x = Random.randn(size(cpws.x))
    x_tangent!(mom_x, cpws, lp)

    # Compute initial Hamiltonian
    hini = Hamiltonian(mom_x, mom_phi, cpws, lp)

    # Molecular Dynamics
    leapfrog!(mom_x, mom_phi, cpws, epsilon, ns, lp)

    # Update Lambda and J_n and compute final Hamiltonian
    sync_fields!(cpws, lp)
    hfin = Hamiltonian(mom_x, mom_phi, cpws, lp)

    dH = hfin - hini
    pacc = exp(-dH)
    if (pacc < 1.0)
        r = rand()
        if (r > pacc) 
            cpws.x .= x_cp
            cpws.phi .= phi_cp
            sync_fields!(cpws, lp)
            @info("    REJECT: Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]")
        else
            @info("    ACCEPT:  Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]")
        end
    else
        @info("    ACCEPT:  Energy [inital: $hini; final: $hfin; difference: $(hfin-hini)]")
    end

    return dH
end

function x_tangent!(mom, cpws::CPworkspace, lp::LattParm)
    x_tangent!(mom, cpws.P_n, cpws.x, cpws, lp)
    return nothing
end

function x_tangent!(mom, P_n, x, cpws::CPworkspace, lp::LattParm)
    for j in 1:lp.iL[1]
        for i in 1:lp.iL[2]
            @inbounds @views P_n .= Matrix(I, 2*lp.N, 2*lp.N) .- x[:,i,j]*transpose(x[:,i,j])
            # @inbounds @views mom[:,i,j] .= P_n * mom[:,i,j]
            @views mul!(mom[:,i,j], P_n, mom[:,i,j])
        end
    end
    return nothing
end

function reversibility!(cpws::CPworkspace, epsilon, ns, lp::LattParm)
    x_cp = copy(cpws.x)
    phi_cp = copy(cpws.phi)

    # Create Lambda and J_n
    sync_fields!(cpws, lp)

    # Create momenta for gauge variables
    mom_phi = Random.randn(size(cpws.phi))

    # Create and project x momenta to tangent space
    mom_x = Random.randn(size(cpws.x))
    x_tangent!(mom_x, cpws, lp)

    leapfrog!(mom_x, mom_phi, cpws, epsilon, ns, lp)
    leapfrog!(-mom_x, -mom_phi, cpws, epsilon, ns, lp)

    dx2 = sum(abs2, cpws.x .- x_cp)
    dphi2 = sum(abs2, cpws.phi .- phi_cp)

    println("|Δx|² = ", dx2)
    println("|Δϕ|² = ", dphi2)

    return nothing
end


function Hamiltonian(mom_x, mom_phi, cpws::CPworkspace, lp::LattParm)
    H = mapreduce(x -> x^2, +, mom_x)/2.0 + mapreduce(x -> x^2, +, mom_phi)/2.0 + action(cpws, lp)
    return H
end

function leapfrog!(mom_x, mom_phi, cpws::CPworkspace, epsilon, ns, lp::LattParm)

	# First half-step for momenta
    update_momenta!(mom_x, mom_phi, cpws, epsilon/2.0, lp)

	# ns-1 steps
	for i in 1:(ns-1) 
		# Update fields
        update_fields!(cpws, mom_x, mom_phi, epsilon, lp) 

		#Update momenta
        update_momenta!(mom_x, mom_phi, cpws, epsilon, lp)
	end
	# Last update for fields
    update_fields!(cpws, mom_x, mom_phi, epsilon, lp) 

	# Last half-step for momenta
    update_momenta!(mom_x, mom_phi, cpws, epsilon/2.0, lp)

	return nothing
end


function update_momenta!(mom_x, mom_phi, cpws::CPworkspace, epsilon, lp::LattParm)

    # Update Lambda and J_n
    sync_fields!(cpws, lp)

    # Load x force and phi force
    load_frcs!(cpws, lp)

    # Update x momenta and phi momenta
    update_momenta!(mom_x, mom_phi, cpws.frc_x, cpws.frc_phi, epsilon, cpws, lp)

    return nothing
end

function update_momenta!(mom_x, mom_phi, frc_x, frc_phi, epsilon, cpws::CPworkspace, lp::LattParm)
    mom_x .= mom_x .+ epsilon .* frc_x
    mom_phi .= mom_phi .+ epsilon .* frc_phi
    return nothing
end

function update_fields!(cpws::CPworkspace, mom_x, mom_phi, epsilon, lp::LattParm) 
    update_fields!(cpws.x, cpws.phi, mom_x, mom_phi, cpws.x_cp, epsilon, cpws, lp)
    return nothing
end

function update_fields!(x, phi, mom_x, mom_phi, x_cp, epsilon, cpws::CPworkspace, lp::LattParm) 
    # Update phi field
    phi .= phi .+ epsilon * mom_phi

    # Update x field
    update_x!(x, mom_x, x_cp, epsilon, cpws, lp)
    project_to_Sn!(x, lp) # project back to Sn, although it should be projected already
    return nothing
end


function update_x!(x, mom_x, x_cp, epsilon, cpws::CPworkspace, lp::LattParm)
    x_cp .= x
    for j in 1:lp.iL[1]
        for i in 1:lp.iL[2]
            @views abspi = sqrt(sum(abs2, mom_x[:,i,j]))
            alpha = epsilon * abspi
            @views x[:,i,j] .= cos(alpha) * x_cp[:,i,j] .+ sin(alpha) * mom_x[:,i,j] / abspi
            @views mom_x[:,i,j] .= - abspi * sin(alpha) * x_cp[:,i,j] .+ cos(alpha) *
                            mom_x[:,i,j]
        end
    end
    return nothing
end
