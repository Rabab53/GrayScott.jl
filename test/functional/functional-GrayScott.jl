import GrayScott
import Test: @testset, @test

function test_grayscott_main(config_file::String, np::Integer=4)
    MPI.mpiexec() do runcmd
        project = Base.active_project()
        juliacmd = `julia --project=$project -e "import GrayScott; GrayScott.main(ARGS)" $config_file`

        @test run(`mpirun -n $np $juliacmd`).exitcode == 0
    end
end
@testset "GrayScott (CPU, Plain)" begin
    test_grayscott_main(joinpath(@__DIR__, "config_cpu_plain.toml"))
end
@testset "GrayScott (CPU, KernelAbstractions)" begin
    test_grayscott_main(joinpath(@__DIR__, "config_cpu_ka.toml"))
end
