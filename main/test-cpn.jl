using Revise, Pkg, TimerOutputsTracked
Pkg.activate(".")
using CPN

# Theory parameters
N = 10
lsize = 42
beta = 0.7
lp0 = LattParm(N, (lsize,lsize), beta)

# Initialize CPN workspace
A0 = CPworkspace(Float64, lp0)

randomize!(A0, lp0)
sync_fields!(A0, lp0)
gauge_frc!(A0, lp0)

x_frc!(A0, lp0)
action(A0, lp0)


# HMC
tau = 1.0
ns = 62
epsilon = tau / ns


@time HMC!(A0, epsilon, ns, lp0)

for i in 1:100000
    dH = HMC!(A0, epsilon, ns, lp0)
    S = action(A0, lp0)

    global io_stat = open("test_output.txt", "a")
    write(io_stat, "$(S),$(dH)\n")
    close(io_stat)
end


# Reversibility

CPN.reversibility!(A0, epsilon, ns, lp0)


# Profiling

struct testruct
    x::Float64
    y::Float64
end

function sum_test(t::testruct)
    return t.x - t.y
end

t = testruct(1.0, 2.0)

TimerOutputsTracked.track([sum_test])

@timetracked sum_test(t)


# Gauge frc test

A0 = CPworkspace(Float64, lp0)
randomize!(A0, lp0)
sync_fields!(A0, lp0)
A2 = deepcopy(A0)

S = action(A0, lp0)

ϵ = 0.000001

A2.phi[2,2,2] += ϵ
sync_fields!(A2, lp0)
S2 = action(A2, lp0)

(S2 - S)/ϵ

gauge_frc!(A0, lp0)

A0.frc_phi[2,2,2]



# x frc test

A0 = CPworkspace(Float64, lp0)
randomize!(A0, lp0)
sync_fields!(A0, lp0)
A2 = deepcopy(A0)

S = action(A0, lp0)

ϵ = 0.0001

A2.x[2,2,2] += ϵ
project_to_Sn!(A2.x, lp0)

sync_fields!(A2, lp0)
S2 = action(A2, lp0)
(S2 - S)/ϵ

x_frc!(A0, lp0)
A0.frc_x[2,2,2]


# Check x_tangent!
# generate cpws and momenta and project them and check orthogonality



# Performance tips
# - slicing an array copies the array. Use @views instead: up to 4 times less
# allocations
# - can use @code_warntype to check type stability
# - it seems that fill_Lambda! cannot infer the output of the cat function, but
# I think that this is just a momentary instability resolved just after, it does
# not propagate, and does not affect much to performance
# - in principle one should avoid structs with abstract containers but I think
# that here it is solved by specializing functions
