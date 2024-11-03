"""
Contains the settings read from the simulation config file (TOML format).
"""
Base.@kwdef mutable struct Settings
    L::Int64 = 128
    steps::Int32 = 20000
    plotgap::Int32 = 200
    F::Float64 = 0.04
    k::Float64 = 0
    dt::Float64 = 0.2
    Du::Float64 = 0.05
    Dv::Float64 = 0.1
    noise::Float64 = 0.0
    output::String = "foo.bp"
    checkpoint::Bool = false
    checkpoint_freq::Int32 = 2000
    checkpoint_output::String = "ckpt.bp"
    restart::Bool = false
    restart_input::String = "ckpt.bp"
    #adios_config::String = "adios2.yaml"
    #adios_span::Bool = false
    #adios_memory_selection::Bool = false
    mesh_type::String = "image"
    precision::String = "Float64"
    backend::String = "CPU"
    kernel_language::String = "Plain"
    verbose::Bool = false
end

# Set of keys that are allowed in the settings file
const SettingsKeys = Set{String}([
    "L",
    "steps",
    "plotgap",
    "F",
    "k",
    "dt",
    "Du",
    "Dv",
    "noise",
    "output",
    "checkpoint",
    "checkpoint_freq",
    "checkpoint_output",
    "restart",
    "restart_input",
    "mesh_type",
    "precision",
    "backend",
    "kernel_language",
    "verbose",
])

"""
Contains the MPI communicator and the Cartesian domain information.
"""
Base.@kwdef mutable struct MPICartDomain
    # MPI Cartesian communicator
    cart_comm::MPI.Comm = MPI.COMM_NULL

    # Dimensions and coordinates of the local process in the Cartesian communicator
    dims::Vector{Int32} = zeros(Int32, 3)
    coords::Vector{Int32} = zeros(Int32, 3)

    # Local process mesh sizes and offsets
    proc_sizes::Vector{Int64} = [128, 128, 128]
    proc_offsets::Vector{Int64} = [1, 1, 1]

    # Neighbor ranks in the MPI Cartesian communicator
    proc_neighbors = Dict{String, Int32}("west" => -1, "east" => -1, "up" => -1,
                                         "down" => -1, "north" => -1,
                                         "south" => -1)
end

"""
Contains the field values (`u` and `v`) to be simulated, as well as the MPI
datatypes used for halo exchange.

The field values are subtype of `AbstractArray` to allow them to be CPU
`Array`s or GPU arrays (like `CuArray` (CUDA) or `ROCArray` (ROCm/AMDGPU)).
"""
mutable struct Fields{T, N, A <: AbstractArray{T, N}}
    # Field values
    u::A
    v::A
    u_temp::A
    v_temp::A

    # MPI datatypes for halo exchange
    xy_face_t::MPI.Datatype
    xz_face_t::MPI.Datatype
    yz_face_t::MPI.Datatype
end
