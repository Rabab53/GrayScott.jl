"""
Initializes the fields for the simulation running on CPU threads.
This function creates two fields, `u` and `v`, representing the concentrations
of two chemical substances over a 3D grid.

# Arguments
- `settings::Settings`: The settings for the simulation
- `mcd::MPICartDomain`: The MPI Cartesian domain
- `T`: The type of the fields (e.g. `Float32`, `Float64`)

# Returns
- `Fields`: The fields for the simulation
"""
function init_fields(::Val{:cpu}, ::Val{:plain}, settings::Settings,
                     mcd::MPICartDomain, T)::Fields{T}
    # Get the size of the local subdomain for the current process
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # Initialize the field for u (chemical U) with 1s and v (chemical V) with 0s.
    # These are 3D arrays with ghost cells (for MPI communication)
    u = ones(T, size_x + 2, size_y + 2, size_z + 2)
    v = zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # Temporary arrays for storing intermediate results
    u_temp = zeros(T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # `d` defines the size of the pattern to initialize in the center of the grid
    d::Int64 = 6

    # Global locations for the central part of the grid
    minL = Int64(settings.L / 2 - d)
    maxL = Int64(settings.L / 2 + d)

    # Process-specific offsets for the current subdomain
    xoff = mcd.proc_offsets[1]
    yoff = mcd.proc_offsets[2]
    zoff = mcd.proc_offsets[3]

    # Initialize a square region in the middle of the domain with specific concentrations
    # This is done in parallel (over the range of `z`) using CPU threads
    Threads.@threads for z in minL:maxL
        for y in minL:maxL
            for x in minL:maxL
                if !is_inside(x, y, z, mcd.proc_offsets, mcd.proc_sizes)
                    continue
                end

                # Set specific values for the chemicals u and v in the central region
                # Julia is 1-indexed, like Fortran
                u[x - xoff + 2, y - yoff + 2, z - zoff + 2] = 0.25
                v[x - xoff + 2, y - yoff + 2, z - zoff + 2] = 0.33
            end
        end
    end

    # Get the faces needed for MPI communication between processes
    xy_face_t, xz_face_t, yz_face_t = get_MPI_faces(size_x, size_y, size_z, T)

    # Return the Fields structure containing u, v, and their temporary arrays
    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

"""
Iterate over the fields for a single time step, updating the concentrations of the
chemical substances `u` and `v` based on the Gray-Scott reaction-diffusion equations.

# Arguments
- `fields::Fields`: The fields containing the concentrations of the chemical substances.
- `settings::Settings`: Simulation settings containing parameters like diffusion rates.
- `mcd::MPICartDomain`: The local subdomain configuration for the current process.

# No return value (operates in-place)
"""
function iterate!(::Val{backend_symbol}, ::Val{:plain},
                  fields::Fields{T, N, Array{T, N}},
                  settings::Settings,
                  mcd::MPICartDomain) where {backend_symbol, T, N}
    # Perform the exchange of ghost cells between neighboring processes
    # This function is communication-bound
    exchange!(fields, mcd)

    # Calculate the new values for the fields `u` and `v` based on the Gray-Scott equations
    # This function is compute/memory-bound
    calculate!(Val{backend_symbol}(), Val{:kernel_abstractions}(), fields, settings, mcd)

    # Swap the fields to prepare for the next iteration
    fields.u, fields.u_temp = fields.u_temp, fields.u
    fields.v, fields.v_temp = fields.v_temp, fields.v

    return
end

"""
Calculates the Gray-Scott reaction-diffusion equations for the fields `u` and `v`
for the current process's local subdomain.

# Arguments:
- `fields::Fields`: The fields containing the concentrations of the chemical substances.
- `settings::Settings`: Simulation settings containing parameters like diffusion rates.

# No return value (operates in-place)
"""
function calculate!(::Val{backend_symbol}, ::Val{:kernel_abstractions},
                    fields::Fields{T, N, Array{T, N}},
                    settings::Settings,
                    mcd::MPICartDomain) where {backend_symbol, T, N}
    # Convert the simulation parameters to the specified type
    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    # Use CPU threads to loop through the local subdomain and calculate the new
    # field values for `u` and `v` based on the Gray-Scott equations
    # The domain of `k` is partitioned across the available CPU threads
    Threads.@threads for k in 2:(mcd.proc_sizes[3] + 1)
        for j in 2:(mcd.proc_sizes[2] + 1)
            @inbounds for i in 2:(mcd.proc_sizes[1] + 1)
                # Get the current values of `u` and `v` at the grid point (i, j, k)
                u = fields.u[i, j, k]
                v = fields.v[i, j, k]

                # Calculate the gradient of `u` and `v` using the Laplacian operator
                # Also introduces a random disturbance on `du`
                du = Du * laplacian(i, j, k, fields.u) - u * v^2 +
                     F * (1.0 - u) +
                     noise * rand(Distributions.Uniform(-1, 1))
                dv = Dv * laplacian(i, j, k, fields.v) + u * v^2 -
                     (F + K) * v

                # Write back the new values to grid point (i, j, k)
                fields.u_temp[i, j, k] = u + du * dt
                fields.v_temp[i, j, k] = v + dv * dt
            end
        end
    end
end

"""
Returns a copy of the fields `u` and `v` without the ghost cells.

# Arguments:
- `fields::Fields`: The fields containing the concentrations of `u` and `v`.

# Returns:
- `u_no_ghost::Array{T, N}`: The field `u` without ghost cells.
- `v_no_ghost::Array{T, N}`: The field `v` without ghost cells.
"""
function get_fields(::Val{:cpu}, fields::Fields{T, N, Array{T, N}}) where {T, N}
    @inbounds begin
        u_no_ghost = fields.u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                              (begin + 1):(end - 1)]
        v_no_ghost = fields.v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                              (begin + 1):(end - 1)]
    end
    return u_no_ghost, v_no_ghost
end
