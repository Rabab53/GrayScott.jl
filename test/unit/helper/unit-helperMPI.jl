
import Test: @testset, @test, @test_throws
import GrayScott.Simulation: Helper

@testset "unit-Helper.bcast_file_contents" begin
    file_contents = Helper.bcast_file_contents(config_file, MPI.COMM_WORLD)
    file_contents_expected = String(read(open(config_file, "r")))
    @test file_contents == file_contents_expected
end
