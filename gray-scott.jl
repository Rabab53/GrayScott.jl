import GrayScott#, ADIOS2
import CUDA
# using Profile
# using PProf

function julia_main()::Cint
    GrayScott.main(ARGS)
    return 0
end

if !isdefined(Base, :active_repl)
    @time julia_main()
    # @profile julia_main()
    # pprof()
end
