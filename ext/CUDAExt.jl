module CUDAExt

import GrayScott
import GrayScott: Simulation
import GrayScott.Simulation: Settings, Fields, MPICartDomain
import Distributions
import CUDA
import CUDA: CuArray

"""
Gets the CUDA backend for kernel acceleration.

# Returns
- `CUDA.CUDABackend()`: The CUDA backend object for GPU computation
"""
Simulation.KA_backend(::Val{:cuda}) = CUDA.CUDABackend()

"""
Initializes the fields for the simulation running on CUDA-enabled GPU.
This function creates two fields, `u` and `v`, representing the concentrations
of two chemical substances over a 3D grid, allocated on the GPU.

# Arguments
- `settings::Settings`: The settings for the simulation
- `mcd::MPICartDomain`: The MPI Cartesian domain
- `T`: The type of the fields (e.g. `Float32`, `Float64`)

# Returns
- `Fields`: The fields for the simulation, stored as CuArrays
"""
function Simulation.init_fields(::Val{:cuda}, ::Val{:plain},
                                settings::Settings,
                                mcd::MPICartDomain,
                                T)::Fields{T, 3, <:CuArray{T, 3}}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # Initialize the field for u (chemical U) with 1s and v (chemical V) with 0s on GPU
    u = CUDA.ones(T, size_x + 2, size_y + 2, size_z + 2)
    v = CUDA.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # Temporary arrays for storing intermediate results on GPU
    u_temp = CUDA.zeros(T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = CUDA.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # Transfer process offsets and sizes to GPU memory
    cu_offsets = CuArray(mcd.proc_offsets)
    cu_sizes = CuArray(mcd.proc_sizes)

    # `d` defines the size of the pattern to initialize in the center of the grid
    d::Int64 = 6
    minL = Int64(settings.L / 2 - d)
    maxL = Int64(settings.L / 2 + d)

    # Configure CUDA kernel execution parameters
    threads = (16, 16)
    blocks = (settings.L, settings.L)

    # Launch kernel to populate the initial values
    CUDA.@cuda threads=threads blocks=blocks populate!(u, v,
                                                       cu_offsets,
                                                       cu_sizes,
                                                       minL, maxL)
    CUDA.synchronize()

    # Get the faces needed for MPI communication between processes
    xy_face_t, xz_face_t, yz_face_t = Simulation.get_MPI_faces(size_x, size_y, size_z, T)

    # Return the Fields structure containing u, v, and their temporary arrays
    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

"""
CUDA kernel function that populates the initial values for the chemical concentrations
in the central region of the domain.

# Arguments
- `u`: Field representing concentration of chemical U
- `v`: Field representing concentration of chemical V
- `offsets`: Process offsets in the global domain
- `sizes`: Sizes of the local subdomain
- `minL`: Minimum coordinate of the central region
- `maxL`: Maximum coordinate of the central region

# No return value (operates in-place)
"""
function populate!(u, v, offsets, sizes, minL, maxL)
    # Calculate local coordinates (1-indexed)
    lz = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
    ly = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y + CUDA.threadIdx().y

    if lz <= size(u, 3) && ly <= size(u, 2)
        # Convert local to global coordinates
        z = lz + offsets[3] - 1
        y = ly + offsets[2] - 1

        if z >= minL && z <= maxL && y >= minL && y <= maxL
            xoff = offsets[1]

            for x in minL:maxL
                # Check if global coordinates for initialization are inside the region
                if !Simulation.is_inside(x, y, z, offsets, sizes)
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
using CUDA-enabled GPU acceleration.

# Arguments
- `fields::Fields`: The fields containing the concentrations of the chemical substances
- `settings::Settings`: Simulation settings containing parameters like diffusion rates
- `mcd::MPICartDomain`: The MPI Cartesian domain information

# No return value (operates in-place)
"""
function Simulation.calculate!(::Val{:cuda}, ::Val{:plain},
                               fields::Fields{T, N, <:CuArray{T, N}},
                               settings::Settings,
                               mcd::MPICartDomain) where {T, N}
    """
    CUDA kernel function that performs the actual calculation of the Gray-Scott
    equations for each grid point.
    """
    function calculate_kernel!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                               noise, dt)
        # Calculate local coordinates (1-indexed)
        k = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        j = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y + CUDA.threadIdx().y

        # Process only non-ghost cells
        if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
            for i in 2:(sizes[1] + 1)
                u_ijk = u[i, j, k]
                v_ijk = v[i, j, k]

                # Calculate the gradient of `u` and `v` using the Laplacian operator
                # Also introduces a random disturbance on `du`
                du = Du * Simulation.laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                     F * (1.0 - u_ijk) +
                     noise * rand(Distributions.Uniform(-1, 1))

                dv = Dv * Simulation.laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
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

    # Transfer process sizes to GPU memory
    cu_sizes = CuArray(mcd.proc_sizes)

    # Configure CUDA kernel execution parameters
    threads = (16, 16)
    blocks = (settings.L, settings.L)

    # Launch kernel to perform calculations
    CUDA.@cuda threads=threads blocks=blocks calculate_kernel!(fields.u,
                                                               fields.v,
                                                               fields.u_temp,
                                                               fields.v_temp,
                                                               cu_sizes,
                                                               Du, Dv, F, K,
                                                               noise, dt)
    CUDA.synchronize()
end

"""
Returns a copy of the fields `u` and `v` without the ghost cells, transferred from GPU to CPU memory.

# Arguments
- `fields::Fields`: The fields containing the concentrations of `u` and `v`

# Returns
- `u_no_ghost::Array{T, N}`: The field `u` without ghost cells
- `v_no_ghost::Array{T, N}`: The field `v` without ghost cells
"""
function Simulation.get_fields(::Val{:cuda}, fields::Fields{T, N, <:CuArray{T, N}}) where {T, N}
    # Transfer data from GPU to CPU and remove ghost cells
    u = Array(fields.u)
    u_no_ghost = u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]

    v = Array(fields.v)
    v_no_ghost = v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]
    return u_no_ghost, v_no_ghost
end

end # module CUDAExt
