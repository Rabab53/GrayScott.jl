"""
GrayScott.jl is a Simulation and Analysis parallel framework for solving the 
Gray-Scott 3D diffusion reaction system of equations of two variables U and V on 
a regular Cartesian mesh.

The bp output files can be visualized with ParaView.
"""

module GrayScott

import MPI #, ADIOS2

using KernelAbstractions #, OffsetArrays


# manages the simulation computation
include(joinpath("simulation", "Simulation.jl"))
import .Simulation

# manages the I/O
#include(joinpath("simulation", "IO.jl"))
#import .IO

function julia_main()::Cint
    try
        main(ARGS)
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

function main(args::Vector{String})

    comm, settings, mpi_cart_domain, fields = Simulation.initialization(args)
    rank = MPI.Comm_rank(comm)
    
    # initialize IOStream struct holding ADIOS-2 components for parallel I/O
    #stream = IO.init(settings, mpi_cart_domain, fields)

    restart_step::Int32 = 0
    # @TODO: checkpoint-restart 
    step::Int32 = restart_step

    while step < settings.steps
        Simulation.iterate!(fields, settings, mpi_cart_domain)
        step += 1

        if step % settings.plotgap == 0
            if rank == 0 && settings.verbose
                println("Simulation at step ", step, " writing output step ",
                        step / settings.plotgap)
            end

           # IO.write_step!(stream, step, fields)
        end
    end

   #IO.close!(stream)

   Simulation.finalize()
end

end # module GrayScott
