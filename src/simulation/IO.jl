"""
Module for handling input/output operations in the simulation using ADIOS2.
This module provides functionality for initializing IO streams and writing
simulation data to disk in a format compatible with visualization tools.
"""
module IO

export init, write_step!

import ADIOS2

import ..Simulation
import ..Simulation: Settings, MPICartDomain, Fields

struct ADIOSStream
    adios::ADIOS2.Adios
    io::ADIOS2.AIO
    engine::ADIOS2.Engine
    var_step::ADIOS2.Variable
    var_U::ADIOS2.Variable
    var_V::ADIOS2.Variable
end

"""
Initializes the ADIOS2 IO system for writing simulation data.
Sets up the necessary variables and attributes for data output,
including visualization schemas for Fides and VTK.

# Arguments
- `settings::Settings`: The simulation settings containing output parameters
- `mcd::MPICartDomain`: MPI Cartesian domain information
- `fields::Fields{T}`: The fields containing simulation data

# Returns
- `ADIOSStream`: An initialized ADIOSStream object for writing data
"""
function init(settings::Settings, mcd::MPICartDomain,
              fields::Fields{T}) where {T}

    # Initialize ADIOS MPI using the cartesian communicator
    adios = ADIOS2.adios_init_mpi(mcd.cart_comm)
    io = ADIOS2.declare_io(adios, "SimulationOutput")
    # TODO: implement ADIOS2.set_engine in ADIOS2.jl
    engine = ADIOS2.open(io, settings.output, ADIOS2.mode_write)

    # Store simulation run provenance as attributes
    # These parameters define the behavior of the Gray-Scott simulation
    ADIOS2.define_attribute(io, "F", settings.F)      # Feed rate
    ADIOS2.define_attribute(io, "k", settings.k)      # Kill rate
    ADIOS2.define_attribute(io, "dt", settings.dt)    # Time step size
    ADIOS2.define_attribute(io, "Du", settings.Du)    # Diffusion rate for U
    ADIOS2.define_attribute(io, "Dv", settings.Dv)    # Diffusion rate for V
    ADIOS2.define_attribute(io, "noise", settings.noise)  # Random noise magnitude

    # Add visualization metadata schemas
    _add_visualization_schemas(io, settings.L)

    # ADIOS2 requires tuples for the dimensions
    # Define global variables u and v with their decomposition
    shape = (settings.L, settings.L, settings.L)     # Global dimensions
    start = Tuple(mcd.proc_offsets)                  # Starting point in global array
    count = Tuple(mcd.proc_sizes)                    # Local array size

    # Define ADIOS2 variables for step counter and chemical concentrations
    var_step = ADIOS2.define_variable(io, "step", Int32)
    var_U = ADIOS2.define_variable(io, "U", T, shape, start, count)
    var_V = ADIOS2.define_variable(io, "V", T, shape, start, count)

    return ADIOSStream(adios, io, engine, var_step, var_U, var_V)
end

"""
Writes the current state of the simulation to disk using ADIOS2.

# Arguments
- `stream::ADIOSStream`: The ADIOS2 IO stream for writing data
- `step::Int32`: The current simulation step number
- `fields::Fields{T}`: The fields containing the chemical concentrations to write

# No return value
"""
function write_step!(stream::ADIOSStream, step::Int32, fields::Fields{T}, settings::Settings) where {T}
    # Get field data without ghost cells for writing
    backend, _ = Simulation.Inputs.load_backend_and_lang(settings)
    u_no_ghost, v_no_ghost = Simulation.get_fields(Val{backend}(), fields)

    # Get writer engine from stream
    w = stream.engine

    # Write data for the current time step
    ADIOS2.begin_step(w)
    ADIOS2.put!(w, stream.var_step, step)
    ADIOS2.put!(w, stream.var_U, u_no_ghost)
    ADIOS2.put!(w, stream.var_V, v_no_ghost)
    ADIOS2.end_step(w)
end

"""
Closes the ADIOS2 IO system, ensuring all data is properly written and resources
are cleaned up.

# Arguments
- `stream::ADIOSStream`: The ADIOS2 IO stream to close

# No return value
"""
function close!(stream::ADIOSStream)
    ADIOS2.close(stream.engine)
    ADIOS2.adios_finalize(stream.adios)
end

"""
Helper function to add visualization metadata schemas for both Fides and VTK.
These schemas enable the simulation output to be visualized using tools like
ParaView.

# Arguments
- `io`: The ADIOS2 IO object to which schemas will be added
- `length`: The size of the simulation domain in each dimension

# No return value
"""
function _add_visualization_schemas(io, length)
    # Fides schema for ParaView
    # Defines a uniform grid with specified origin, spacing, and variables
    ADIOS2.define_attribute(io, "Fides_Data_Model", "uniform")
    ADIOS2.define_attribute_array(io, "Fides_Origin", [0.0, 0.0, 0.0])
    ADIOS2.define_attribute_array(io, "Fides_Spacing", [0.1, 0.1, 0.1])
    ADIOS2.define_attribute(io, "Fides_Dimension_Variable", "U")
    ADIOS2.define_attribute_array(io, "Fides_Variable_List", ["U", "V"])
    ADIOS2.define_attribute_array(io, "Fides_Variable_Associations",
                                  ["points", "points"])

    # VTK XML schema
    # String concatenation uses *, ^ is for repetition
    # Example: if length = 64, extent = "0 64 0 64 0 64 "
    extent = ("0 " * string(length) * " ")^3
    extent = rstrip(extent)

    # Deactivate code formatting using JuliaFormatter.jl
    # Raw strings: " must be escaped with \"
    #! format: off
    vtx_schema = raw"
        <?xml version=\"1.0\"?>
        <VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">
          <ImageData WholeExtent=\"" * extent * raw"\" Origin=\"0 0 0\" Spacing=\"1 1 1\">
            <Piece Extent=\"" * extent * raw"\">
              <CellData Scalars=\"U\">
                <DataArray Name=\"U\" />
                <DataArray Name=\"V\" />
                <DataArray Name=\"TIME\">
                  step
                </DataArray>
              </CellData>
            </Piece>
          </ImageData>
        </VTKFile>"
    #! format: on
    # Reactivate code formatting using JuliaFormatter.jl

    # Add the VTK XML schema as an attribute
    ADIOS2.define_attribute(io, "vtk.xml", vtx_schema)
end

end # module IO
