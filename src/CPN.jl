module CPN

import Random
using LinearAlgebra

struct LattParm
    N::Int64
    iL::Tuple{Int64,Int64}
    beta::Float64
end
export LattParm

include("fields.jl")
export CPworkspace, randomize!, fill_Lambda!, fill_Jn!, sync_fields!, project_to_Sn!

include("action.jl")
export action, gauge_frc!, x_frc!, load_frcs!

include("hmc.jl")
export HMC!


end # module


# Glossary of variable name meanings

# ws = workspace
# lp = lattice parameter
# frc = force

