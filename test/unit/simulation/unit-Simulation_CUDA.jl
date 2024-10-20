
import Test: @testset, @test, @test_throws
# import submodule
import GrayScott: Simulation
# import types
import .Simulation: Settings, MPICartDomain, Fields

using CUDA

@testset "unit-Simulation.Init_Fields-cuda" begin
    function test_init_cuda(L)
        settings = Settings()
        settings.L = L
        mpi_cart_domain = Simulation.Init_Domain(settings, MPI.COMM_WORLD)

        fields = Simulation.Init_Fields(settings, mpi_cart_domain, Float32)

        settings.backend = "CUDA"
        fields_cuda = Simulation.Init_Fields(settings, mpi_cart_domain, Float32)

        @test fields.u ≈ Array(fields_cuda.u)
        @test fields.v ≈ Array(fields_cuda.v)
    end

    if CUDA.functional()
        test_init_cuda(8)
        test_init_cuda(16)
        test_init_cuda(32)
        test_init_cuda(64)
        test_init_cuda(128)
    end
end
