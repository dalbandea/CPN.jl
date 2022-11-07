using Revise, Pkg, TimerOutputsTracked
Pkg.activate(".")
using CPN

# Theory parameters
N = 2
lsize = 10
beta = 1.0
lp0 = LattParm(N, (lsize,lsize), beta)

# Initialize CPN workspace
A0 = CPworkspace(Float64, lp0)

randomize!(A0, lp0)
sync_fields!(A0, lp0)
gauge_frc!(A0, lp0)
x_frc!(A0, lp0)
action(A0, lp0)


# HMC
acc = Vector{Int64}()
tau = 1.0
ns = 20
epsilon = tau / ns


@time HMC!(A0, epsilon, ns, acc, lp0)

for i in 1:10000
    HMC!(A0, epsilon, ns, acc, lp0)
end

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

ϵ = 0.00001

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
