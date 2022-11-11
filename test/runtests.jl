using Test

@testset verbose = true "Basic CPN" begin
    @testset verbose = true "Forces" begin
        include("force-test.jl")
    end
end
