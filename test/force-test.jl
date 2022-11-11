using CPN

# Theory parameters
N = 2
lsize = 12
beta = 1.1
lp0 = LattParm(N, (lsize,lsize), beta)

ϵ = 0.000001

# Gauge frc test

A0 = CPworkspace(Float64, lp0)
randomize!(A0, lp0)
sync_fields!(A0, lp0)
S0 = action(A0, lp0)
gauge_frc!(A0, lp0)

F_diff = 0.0
for j in 1:lp0.iL[2], i in 1:lp0.iL[1], mu in 1:2
    A2 = deepcopy(A0)
    A2.phi[mu,i,j] += ϵ
    sync_fields!(A2, lp0)
    S2 = action(A2, lp0)
    Fdiff = (S2 - S0)/ϵ + A2.frc_phi[mu,i,j] |> abs
    @test F_diff ≈ 0.0 atol = 0.0001
end


# @testset "Gauge force" begin
#     @test F_num + F_ana |> abs ≈ 0.0 atol = 0.0001
# end


# x frc test

A2 = deepcopy(A0)
A2.x[2,2,2] += ϵ
project_to_Sn!(A2.x, lp0)
sync_fields!(A2, lp0)
S2 = action(A2, lp0)


F_num = (S2 - S0)/ϵ

x_frc!(A0, lp0)
F_ana = A0.frc_x[2,2,2]


@testset "x force" begin
    @test F_num + F_ana |> abs ≈ 0.0 atol = 0.0001
end
