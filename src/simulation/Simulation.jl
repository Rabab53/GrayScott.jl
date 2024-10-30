""" 
The present file contains runtime backend for using CPU Threads, and optionally 
CUDA.jl, AMDGPU.jl, and KernelAbstractions.jl
"""
module Simulation

export init_domain, init_fields, get_fields

import MPI
import Distributions

# contains relevant data containers "structs" for Input, Domain and Fields
include("Structs.jl")
import .Settings, .MPICartDomain, .Fields

# initializes inputs from configuration file
include("Inputs.jl")
import .Inputs


include("Common.jl")

# include functions for CPU multithreading
include("Simulation_CPU.jl")
# include functions for KernelAbstractions execution (CPU/GPU)
include("Simulation_KA.jl")

# include functions for MPI communication
include("communication.jl")

# include public functions to setup and drive a simulation
include("public.jl")

include("IO.jl")
import .IO

end # module
