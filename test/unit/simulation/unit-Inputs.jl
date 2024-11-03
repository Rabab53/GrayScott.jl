
import Test: @testset, @test, @test_throws
import GrayScott.Simulation: Inputs

# Unfortunately due to MPI being a Singleton, single MPI.Init()
# these unit tests don't run as independent files

@testset "unit-Inputs.get_settings" begin
    Inputs.get_settings([config_file])

    @test_throws(ArgumentError, Inputs.get_settings(["hello.nojson"]))
end
