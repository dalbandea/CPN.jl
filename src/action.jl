
function action(cpws::CPworkspace, lp::LattParm)
    return action(cpws.x, cpws.J_n, cpws, lp)
end

function action(x, J_n, cpws::CPworkspace, lp::LattParm)
    xJ = zero(eltype(x))
    for i in 1:lp.iL[1]
        for j in 1:lp.iL[2]
            xJ += transpose(x[i,j,:]) * J_n[i,j,:]
        end
    end
    return -lp.N * lp.beta * (xJ - 4 * lp.iL[1] * lp.iL[2])
end

function gauge_frc!(cpws::CPworkspace, lp::LattParm)
    gauge_frc!(cpws.frc_phi, cpws.x, cpws.Lambda, cpws.Gamma, cpws, lp)
end

function gauge_frc!(frc_phi, x, Lambda, Gamma, cpws::CPworkspace, lp::LattParm)
    for i in 1:lp.iL[1]
        for j in 1:lp.iL[2]
            for mu in 1:2
                iu = ((i+(mu==1)) - 1) % lp.iL[1] + 1
                ju = ((j+(mu==2)) - 1) % lp.iL[2] + 1
                frc_phi[i,j,mu] = -2 * lp.N * lp.beta * transpose(x[i,j,:]) * Gamma * transpose(Lambda[i,j,mu,:,:]) * x[iu,ju,:]
            end
        end
    end
end

function x_frc!(cpws::CPworkspace, lp::LattParm)
    x_frc!(cpws.frc_x, cpws.P_n, cpws.x, cpws.J_n, cpws, lp)
end

function x_frc!(frc_x, P_n, x, J_n, cpws::CPworkspace, lp::LattParm)
    for i in 1:lp.iL[1]
        for j in 1:lp.iL[2]
            P_n .= Matrix(I, 2*lp.N, 2*lp.N) .- x[i,j,:]*transpose(x[i,j,:])
            frc_x[i,j,:] .= 2 * lp.N * lp.beta * P_n * J_n[i,j,:]
        end
    end
end

function load_frcs!(cpws::CPworkspace, lp::LattParm)
    gauge_frc!(cpws, lp)
    x_frc!(cpws, lp)
    return nothing
end
