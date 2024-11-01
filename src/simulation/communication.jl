"""
Initializes MPI, parses the configuration file, and initializes the domain and
fields. The results of calling this function are sufficient to begin the
simulation.

# Arguments:
- `args::Vector{String}`: Command-line arguments passed to the program.

# Returns:
- `MPI.Comm`: The MPI communicator used for the simulation.
- `Settings`: The simulation settings read from the configuration file.
- `MPICartDomain`: The MPI Cartesian domain used for halo exchange.
- `Fields`: The fields `u` and `v` that the simulation will evolve.
"""
function initialization(args::Vector{String}=ARGS)
    # Parse the configuration file
    settings = Inputs.get_settings(args)

    # Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD

    # Initialize MPI Cartesian domain
    mpi_cart_domain = init_domain(settings, comm)

    # Parse the sepecified precision into a Julia data type
    T = eval(Meta.parse(settings.precision))

    # Initialize fields (`u` and `v`) for the simulation
    fields = init_fields(settings, mpi_cart_domain, T)

    return comm, settings, mpi_cart_domain, fields
end

"""
Finalize MPI if not running in an interactive session.

# No return value
"""
function finalize()
    # Only useful when run in a Julia REPL
    # Not required when run as a script, as MPI is finalized on exit
    if !isinteractive()
        MPI.Finalize()
    end
end

"""
Initialize the domain for the Gray-Scott simulation, setting up the MPI
Cartesian communicator and determining local subdomain sizes.

# Arguments:
- `settings::Settings`: Simulation settings containing the global domain size.
- `comm::MPI.Comm`: The MPI communicator to use.

# Returns:
- `MPICartDomain`: An object containing information about the local subdomain.
"""
function init_domain(settings::Settings, comm::MPI.Comm)::MPICartDomain
    # Create an MPICartDomain object to store domain information
    mcd = MPICartDomain()

    # Set dimensions and Cartesian communicator
    mcd.dims = MPI.Dims_create(MPI.Comm_size(comm), mcd.dims)
    mcd.cart_comm = MPI.Cart_create(comm, mcd.dims)

    # Set process-local coordinates in the Cartesian communicator
    rank = MPI.Comm_rank(comm)
    mcd.coords = MPI.Cart_coords(mcd.cart_comm, rank)

    # set proc local mesh sizes
    # Set process-local mesh sizes based on global domain size `L` and grid dimensions
    mcd.proc_sizes = settings.L ./ mcd.dims

    # Adjust sizes if some dimensions are not evenly divisible
    for (i, coord) in enumerate(mcd.coords)
        if coord < settings.L % mcd.dims[i]
            mcd.proc_sizes[i] += 1
        end
    end

    # Set the offsets for each process' local subdomain
    for i in 1:3
        mcd.proc_offsets[i] = settings.L / mcd.dims[i] *
                              mcd.coords[i] +
                              min(settings.L % mcd.dims[i], mcd.coords[i])
    end

    # Get the ranks of neighboring processes for MPI communication
    mcd.proc_neighbors["west"], mcd.proc_neighbors["east"] = MPI.Cart_shift(mcd.cart_comm, 0, 1)
    mcd.proc_neighbors["down"], mcd.proc_neighbors["up"] = MPI.Cart_shift(mcd.cart_comm, 1, 1)
    mcd.proc_neighbors["south"], mcd.proc_neighbors["north"] = MPI.Cart_shift(mcd.cart_comm, 2, 1)


    return mcd
end

"""
Helper function to get the MPI datatypes for exchanging faces between processes.

# Arguments:
- `size_x::Int`: The size of the local subdomain in the x-direction.
- `size_y::Int`: The size of the local subdomain in the y-direction.
- `size_z::Int`: The size of the local subdomain in the z-direction.

# Returns:
- `(xy_face_t, xz_face_t, yz_face_t)`: A `Tuple` of MPI datatypes for exchanging faces.
"""
function get_MPI_faces(size_x, size_y, size_z, T)
    # Create a new MPI vector type taking: count, block length, stride to
    # interoperate with MPI for ghost cell exchange
    xy_face_t = MPI.Types.create_vector(size_y + 2, size_x, size_x + 2,
                                        MPI.Datatype(T))
    xz_face_t = MPI.Types.create_vector(size_z, size_x,
                                        (size_x + 2) * (size_y + 2),
                                        MPI.Datatype(T))
    yz_face_t = MPI.Types.create_vector((size_y + 2) * (size_z + 2), 1,
                                        size_x + 2, MPI.Datatype(T))

    # Commit the new MPI face types
    MPI.Types.commit!(xy_face_t)
    MPI.Types.commit!(xz_face_t)
    MPI.Types.commit!(yz_face_t)

    return xy_face_t, xz_face_t, yz_face_t
end

"""
Exchange ghost cells between neighboring processes to ensure that each process has
the correct boundary values for the fields `u` and `v`.

# Arguments:
- `fields::Fields`: The fields containing the concentrations of the chemical substances.
- `mcd::MPICartDomain`: The local subdomain configuration for the current process.

# No return value (operates in-place)
"""
function exchange!(fields, mcd)
    # Define some internal functions for exchanging faces between processes

    "Send XY face z=size_z+1 to north and receive z=1 from south."
    function exchange_xy!(var, size_z, data_type, rank1, rank2, comm)
        # To north
        send_buf = MPI.Buffer(@view(var[2, 1, size_z + 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # To south
        send_buf = MPI.Buffer(@view(var[2, 1, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, size_z + 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    "Send XZ face y=size_y+1 to up and receive y=1 from down."
    function exchange_xz!(var, size_y, data_type, rank1, rank2, comm)
        # To up
        send_buf = MPI.Buffer(@view(var[2, size_y + 1, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # To down
        send_buf = MPI.Buffer(@view(var[2, 2, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, size_y + 2, 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    "Send YZ face x=size_x+2 to east and receive x=2 from west."
    function exchange_yz!(var, size_x, data_type, rank1, rank2, comm)
        # To east
        send_buf = MPI.Buffer(@view(var[size_x + 1, 1, 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[1, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # To west
        send_buf = MPI.Buffer(@view(var[2, 1, 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[size_x + 2, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    # If `u` and `v` are not on the CPU, make a copy on the CPU
    # `Array` is a CPU-only type, and automatically performs a D-to-H copy
    u = typeof(fields.u) <: Array ? fields.u : Array(fields.u)
    v = typeof(fields.v) <: Array ? fields.v : Array(fields.v)

    # Exchange ghost cells for `u` and `v` between neighboring processes
    for var in [u, v]
        exchange_xy!(var, mcd.proc_sizes[3], fields.xy_face_t,
                     mcd.proc_neighbors["north"], mcd.proc_neighbors["south"],
                     mcd.cart_comm)

        exchange_xz!(var, mcd.proc_sizes[2], fields.xz_face_t,
                     mcd.proc_neighbors["up"], mcd.proc_neighbors["down"],
                     mcd.cart_comm)

        exchange_yz!(var, mcd.proc_sizes[1], fields.yz_face_t,
                     mcd.proc_neighbors["east"], mcd.proc_neighbors["west"],
                     mcd.cart_comm)
    end
end
