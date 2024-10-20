""" 
The present file contains runtime backend for using CPU Threads, and optionally 
CUDA.jl, AMDGPU.jl, and KernelAbstractions.jl
"""
module Simulation

export Init_Domain, Init_Fields

import MPI
import Distributions

# contains relevant data containers "structs" for Input, Domain and Fields
include("Structs.jl")
import .Settings, .MPICartDomain, .Fields

# contains helper functions for general use
include("../helper/Helper.jl")
import .Helper

# initializes inputs from configuration file
include("Inputs.jl")
import .Inputs

include("Common.jl")


# include functions for CPU multithreading
include("Simulation_CPU.jl")
# include functions for NVIDIA GPUs using CUDA.jl
#include("Simulation_CUDA.jl")
# include functions for AMD GPUs using AMDGPU.jl
#include("Simulation_AMDGPU.jl")
# include functions for all GPUs using KernelAbstractions.jl
include("Simulation_KA.jl")

include("communication.jl")

end # module