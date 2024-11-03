import KernelAbstractions
import KernelAbstractions: @kernel, @index

"""
Gets the CPU backend for KernelAbstractions.

# Returns
- `KernelAbstractions.CPU()`: The CPU backend for kernel execution
"""
KA_backend(::Val{:cpu}) = KernelAbstractions.CPU()

"""
Fallback method for unknown backends.

# Arguments
- `backend_symbol`: The symbol representing the requested backend

# Throws
- Error with message indicating unknown backend
"""
KA_backend(::Val{backend_symbol}) where {backend_symbol} =
    error("Unknown backend: $backend_symbol")

"""
Helper function to allocate and copy data for KernelAbstractions arrays.
This is necessary since KernelAbstractions doesn't provide a built-in
allocate-and-copy function.

# Arguments
- `Backend`: The KernelAbstractions backend to use
- `A`: The source array to copy from

# Returns
- A new array of the same size and type as A, allocated on the specified backend
"""
function similar_KA(Backend, A)
    B = KernelAbstractions.allocate(Backend, eltype(A), size(A))
    KernelAbstractions.copyto!(Backend, B, A)
    return B
end

"""
Initializes the fields for the simulation running on any backend, using KernelAbstractions.
This function creates two fields, `u` and `v`, representing the concentrations
of two chemical substances over a 3D grid.

# Arguments
- `settings::Settings`: The settings for the simulation
- `mcd::MPICartDomain`: The MPI Cartesian domain
- `T`: The type of the fields (e.g. `Float32`, `Float64`)

# Returns
- `Fields`: The fields for the simulation, allocated using KernelAbstractions
"""
function init_fields(::Val{backend_symbol}, ::Val{:kernelabstractions},
                     settings::Settings,
                     mcd::MPICartDomain,
                     T)::Fields{T, 3} where {backend_symbol}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # Get the appropriate backend for kernel execution
    Backend = KA_backend(Val{backend_symbol}())

    # Initialize the field for u (chemical U) with 1s and v (chemical V) with 0s
    u = KernelAbstractions.ones(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)

    # Temporary arrays for storing intermediate results
    u_temp = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)

    # Transfer process offsets and sizes to the backend
    cu_offsets = similar_KA(Backend, mcd.proc_offsets)
    cu_sizes = similar_KA(Backend, mcd.proc_sizes)

    # `d` defines the size of the pattern to initialize in the center of the grid
    d::Int64 = 6
    minL = Int32(settings.L / 2 - d)
    maxL = Int32(settings.L / 2 + d)

    # Configure kernel execution parameters
    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])

    # Launch kernel to populate the initial values
    kernel! = populate_kernel!(Backend, threads)
    kernel!(u, v,
            cu_offsets,
            cu_sizes,
            minL, maxL;
            ndrange=nrange)

    # Ensure kernel execution is complete
    KernelAbstractions.synchronize(Backend)

    # Get the faces needed for MPI communication between processes
    xy_face_t, xz_face_t, yz_face_t = get_MPI_faces(size_x, size_y, size_z, T)

    # Return the Fields structure containing u, v, and their temporary arrays
    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

"""
KernelAbstractions kernel function that populates the initial values for the chemical
concentrations in the central region of the domain.

# Arguments
- `u`: Field representing concentration of chemical U
- `v`: Field representing concentration of chemical V
- `offsets`: Process offsets in the global domain
- `sizes`: Sizes of the local subdomain
- `minL`: Minimum coordinate of the central region
- `maxL`: Maximum coordinate of the central region

# No return value (operates in-place)

Note: Unlike CUDA/ROCm, KernelAbstractions uses @index(Global, NTuple) for thread indexing
"""
@kernel function populate_kernel!(u, v, offsets, sizes, minL, maxL)
    # Calculate local coordinates using KernelAbstractions indexing
    # This replaces the CUDA/ROCm-style block and thread indexing
    lz, ly = @index(Global, NTuple)

    if lz <= size(u, 3) && ly <= size(u, 2)
        # Convert local to global coordinates
        z = lz + offsets[3] - 1
        y = ly + offsets[2] - 1

        if z >= minL && z <= maxL && y >= minL && y <= maxL
            xoff = offsets[1]

            for x in minL:maxL
                # Check if global coordinates for initialization are inside the region
                if !is_inside(x, y, z, offsets, sizes)
                    continue
                end

                # Set specific values for the chemicals u and v in the central region
                u[x - xoff + 2, ly + 1, lz + 1] = 0.25
                v[x - xoff + 2, ly + 1, lz + 1] = 0.33
            end
        end
    end
end

"""
Calculates the Gray-Scott reaction-diffusion equations for the fields `u` and `v`
using KernelAbstractions for the current process's local subdomain.

# Arguments
- `fields::Fields`: The fields containing the concentrations of the chemical substances
- `settings::Settings`: Simulation settings containing parameters like diffusion rates
- `mcd::MPICartDomain`: The MPI Cartesian domain information

# No return value (operates in-place)
"""
function calculate!(::Val{backend_symbol}, ::Val{:kernelabstractions},
                    fields::Fields{T, N},
                    settings::Settings,
                    mcd::MPICartDomain) where {backend_symbol, T, N}
    """
    KernelAbstractions kernel function that performs the actual calculation of the
    Gray-Scott equations for each grid point.

    # Arguments
    - `u`, `v`: Current state of the chemical concentrations
    - `u_temp`, `v_temp`: Temporary arrays for storing the next state
    - `sizes`: Sizes of the local subdomain
    - `Du`, `Dv`: Diffusion coefficients for chemicals U and V
    - `F`, `K`: Feed and kill rates for the Gray-Scott model
    - `noise`: Magnitude of random fluctuations
    - `dt`: Time step size
    """
    @kernel function calculate_kernel!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                                       noise, dt)
        # Get thread indices using KernelAbstractions indexing
        k, j = @index(Global, NTuple)

        # Process only non-ghost cells
        if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
            # Bounds are inclusive
            for i in 2:(sizes[1] + 1)
                u_ijk = u[i, j, k]
                v_ijk = v[i, j, k]

                # Calculate the gradient of `u` and `v` using the Laplacian operator
                # Also introduces a random disturbance on `du`
                du = Du * laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                     F * (1.0 - u_ijk) +
                     noise * rand(Distributions.Uniform(-1, 1))

                dv = Dv * laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
                     (F + K) * v_ijk

                # Write back the new values to grid point (i, j, k)
                u_temp[i, j, k] = u_ijk + du * dt
                v_temp[i, j, k] = v_ijk + dv * dt
            end
        end
    end

    # Convert simulation parameters to the specified type
    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    # Get the appropriate backend for kernel execution
    Backend = KA_backend(Val{backend_symbol}())

    # Transfer process sizes to the backend
    cu_sizes = similar_KA(Backend, mcd.proc_sizes)

    # Configure kernel execution parameters
    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])

    # Launch kernel to perform calculations
    kernel! = calculate_kernel!(Backend, threads)
    kernel!(fields.u,
            fields.v,
            fields.u_temp,
            fields.v_temp,
            cu_sizes,
            Du, Dv, F, K,
            noise, dt,
            ndrange=nrange)

    # Ensure kernel execution is complete
    KernelAbstractions.synchronize(Backend)
end
