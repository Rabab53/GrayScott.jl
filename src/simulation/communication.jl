function initialization(args::Vector{String})

    MPI.Init()
    comm = MPI.COMM_WORLD

    # a data struct that holds settings data from config_file in args
    # example config file: ../examples/settings-files.json
    settings = Inputs.get_settings(args, comm)

    # initialize MPI Cartesian Domain and Communicator
    mpi_cart_domain = init_domain(settings, comm)

    # initialize fields
    fields = init_fields(settings,
                         mpi_cart_domain,
                         Helper.get_type(settings.precision))

    return comm, settings, mpi_cart_domain, fields

end

function finalize()
    # Debugging session or Julia REPL session, not needed overall as it would be 
    # called when the program ends
    if !isinteractive()
        MPI.Finalize()
    end
end

function init_domain(settings::Settings, comm::MPI.Comm)::MPICartDomain
    mcd = MPICartDomain()

    # set dims and Cartesian communicator
    mcd.dims = MPI.Dims_create(MPI.Comm_size(comm), mcd.dims)
    mcd.cart_comm = MPI.Cart_create(comm, mcd.dims)

    # set proc local coordinates in Cartesian communicator
    rank = MPI.Comm_rank(comm)
    mcd.coords = MPI.Cart_coords(mcd.cart_comm, rank)

    # set proc local mesh sizes
    mcd.proc_sizes = settings.L ./ mcd.dims

    for (i, coord) in enumerate(mcd.coords)
        if coord < settings.L % mcd.dims[i]
            mcd.proc_sizes[i] += 1
        end
    end

    # set proc local offsets
    for i in 1:3
        mcd.proc_offsets[i] = settings.L / mcd.dims[i] *
                              mcd.coords[i]
        +min(settings.L % mcd.dims[i], mcd.coords[i])
    end

    # get neighbors ranks
    mcd.proc_neighbors["west"],
    mcd.proc_neighbors["east"] = MPI.Cart_shift(mcd.cart_comm, 0, 1)
    mcd.proc_neighbors["down"],
    mcd.proc_neighbors["up"] = MPI.Cart_shift(mcd.cart_comm, 1, 1)
    mcd.proc_neighbors["south"],
    mcd.proc_neighbors["north"] = MPI.Cart_shift(mcd.cart_comm, 2, 1)

    return mcd
end

"""
Create and Initialize fields for either CPU, CUDA.jl, AMDGPU.jl backends
Multiple dispatch would direct to the appropriate overleaded function
"""
function init_fields(settings::Settings,
                     mcd::MPICartDomain, T)::Fields{T}
    lowercase_backend = lowercase(settings.backend)
    if lowercase_backend == "cuda"
        return init_fields_CUDA(settings, mcd, T)
    elseif lowercase_backend == "amdgpu"
        return init_fields_AMDGPU(settings, mcd, T)
    end
    # everything else would trigger the CPU threads backend
    return init_fields_CPU(settings, mcd, T)
end

function get_MPI_faces(size_x, size_y, size_z, T)

    ## create a new type taking: count, block length, stride
    ## to interoperate with MPI for ghost cell exchange
    xy_face_t = MPI.Types.create_vector(size_y + 2, size_x, size_x + 2,
                                        MPI.Datatype(T))
    xz_face_t = MPI.Types.create_vector(size_z, size_x,
                                        (size_x + 2) * (size_y + 2),
                                        MPI.Datatype(T))
    yz_face_t = MPI.Types.create_vector((size_y + 2) * (size_z + 2), 1,
                                        size_x + 2, MPI.Datatype(T))
    MPI.Types.commit!(xy_face_t)
    MPI.Types.commit!(xz_face_t)
    MPI.Types.commit!(yz_face_t)

    return xy_face_t, xz_face_t, yz_face_t
end

function exchange!(fields, mcd)
    """
    Send XY face z=size_z+1 to north and receive z=1 from south
    """
    function exchange_xy!(var, size_z, data_type, rank1, rank2, comm)
        # to north
        send_buf = MPI.Buffer(@view(var[2, 1, size_z + 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # to south
        send_buf = MPI.Buffer(@view(var[2, 1, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, size_z + 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    """
    Send XZ face y=size_y+1 to up and receive y=1 from down
    """
    function exchange_xz!(var, size_y, data_type, rank1, rank2, comm)
        # to up
        send_buf = MPI.Buffer(@view(var[2, size_y + 1, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, 1, 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # to down
        send_buf = MPI.Buffer(@view(var[2, 2, 2]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[2, size_y + 2, 2]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    """
    Send YZ face x=size_x+2 to east and receive x=2 from west
    """
    function exchange_yz!(var, size_x, data_type, rank1, rank2, comm)
        # to east
        send_buf = MPI.Buffer(@view(var[size_x + 1, 1, 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[1, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank1, source = rank2)

        # to west
        send_buf = MPI.Buffer(@view(var[2, 1, 1]), 1, data_type)
        recv_buf = MPI.Buffer(@view(var[size_x + 2, 1, 1]), 1, data_type)
        MPI.Sendrecv!(send_buf, recv_buf, comm, dest = rank2, source = rank1)
    end

    # if already a CPU array, no need to copy,
    # otherwise (device) copy to host.
    u = typeof(fields.u) <: Array ? fields.u : Array(fields.u)
    v = typeof(fields.v) <: Array ? fields.v : Array(fields.v)

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
