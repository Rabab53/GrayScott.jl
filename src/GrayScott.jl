"""
GrayScott.jl - Parallel Simulation Framework for Gray-Scott Reaction-Diffusion System

A parallel framework for simulating and analyzing the Gray-Scott 3D reaction-diffusion
system, which models the interaction between two chemical species U and V on a regular
Cartesian mesh. The system demonstrates pattern formation and self-organization in
chemical systems.

The simulation uses:
- MPI for parallel computation across multiple processes
- KernelAbstractions for hardware-agnostic kernel acceleration
- ADIOS2 for parallel I/O (when enabled)

Output files are written in BP format, compatible with ParaView for visualization.

# Example Usage
```julia
using GrayScott
GrayScott.main(["settings.json"])  # Run simulation with settings file
```
"""
module GrayScott

import MPI
import ADIOS2

using KernelAbstractions #, OffsetArrays

# Include core simulation module
include(joinpath("simulation", "Simulation.jl"))
import .Simulation

"""
Entry point for running GrayScott as an executable.
Provides error handling and returns appropriate exit codes.

# Returns
- `Cint`: 0 for successful execution, 1 if an error occurred
"""
function julia_main()::Cint
    try
        main(ARGS)
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

"""
Main simulation driver function that coordinates the simulation execution.

# Arguments
- `args::Vector{String}`: Command line arguments, typically containing path to settings file

The function performs these key steps:
1. Initializes MPI, simulation settings, and fields
2. Sets up parallel I/O if enabled
3. Runs the main simulation loop
4. Periodically writes output for visualization
5. Cleans up resources on completion

# Example
```julia
main(["settings.json"])
```
"""
function main(args::Vector{String})
    # Initialize MPI, simulation settings, domain decomposition, and fields
    comm, settings, mpi_cart_domain, fields = Simulation.initialization(args)
    rank = MPI.Comm_rank(comm)

    # Initialize ADIOS2 I/O stream for parallel output
    stream = Simulation.IO.init(settings, mpi_cart_domain, fields)

    # Initialize simulation step counters
    restart_step::Int32 = 0
    step::Int32 = restart_step

    # Main simulation loop
    while step < settings.steps
        # Compute next time step
        Simulation.iterate!(fields, settings, mpi_cart_domain)
        step += 1

        # Periodically write output for visualization
        if step % settings.plotgap == 0
            if rank == 0 && settings.verbose
                println("Simulation at step ", step, " writing output step ",
                        step / settings.plotgap)
            end

            # Write visualization data
            Simulation.IO.write_step!(stream, step, fields, settings)
        end
    end

    # Clean up I/O resources
    Simulation.IO.close!(stream)

    # Clean up simulation resources
    Simulation.finalize()
end

end # module GrayScott
