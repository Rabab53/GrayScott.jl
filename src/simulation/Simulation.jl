module Simulation

import MPI
import Distributions

# Defines the data structures used in the simulation
include("Structs.jl")
import .Settings, .MPICartDomain, .Fields

# Defines functions to load and parse the configuration file
include("Inputs.jl")
import .Inputs

# Common functions used by the simulation internally
include("Common.jl")

# Defines functions for CPU multithreading with Julia's Threads module
include("Simulation_CPU.jl")
# Defines functions for CPU/GPU acceleration with KernelAbstractions.jl
include("Simulation_KA.jl")

# Defines functions for MPI communication and halo exchange
include("communication.jl")

# Defines public functions to setup and drive a GrayScott simulation
include("public.jl")

end # module
