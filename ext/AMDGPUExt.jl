module AMDGPUExt

import GrayScott
import GrayScott: Simulation
import GrayScott.Simulation: Settings, Fields, MPICartDomain
import AMDGPU
import AMDGPU: ROCArray

"""
Gets the AMD GPU (ROCm) backend for kernel acceleration.

# Returns
- `AMDGPU.ROCBackend()`: The ROCm backend object for AMD GPU computation
"""
Simulation.KA_backend(::Val{:amdgpu}) = AMDGPU.ROCBackend()

"""
Initializes the fields for the simulation running on AMD GPU using ROCm.
This function creates two fields, `u` and `v`, representing the concentrations
of two chemical substances over a 3D grid, allocated on the AMD GPU.

# Arguments
- `settings::Settings`: The settings for the simulation
- `mcd::MPICartDomain`: The MPI Cartesian domain
- `T`: The type of the fields (e.g. `Float32`, `Float64`)

# Returns
- `Fields`: The fields for the simulation, stored as ROCArrays
"""
function Simulation.init_fields(::Val{:amdgpu}, ::Val{:plain},
                                settings::Settings,
                                mcd::MPICartDomain,
                                T)::Fields{T, 3, <:ROCArray{T, 3}}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # Initialize the field for u (chemical U) with 1s and v (chemical V) with 0s on GPU
    u = AMDGPU.ones(T, size_x + 2, size_y + 2, size_z + 2)
    v = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # Temporary arrays for storing intermediate results on GPU
    u_temp = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    # Transfer process offsets and sizes to GPU memory
    roc_offsets = ROCArray(mcd.proc_offsets)
    roc_sizes = ROCArray(mcd.proc_sizes)

    # `d` defines the size of the pattern to initialize in the center of the grid
    d::Int64 = 6
    minL = Int64(settings.L / 2 - d)
    maxL = Int64(settings.L / 2 + d)

    # Configure ROCm kernel execution parameters
    threads = (16, 16)
    # Grid size must be the total number of threads in each direction
    grid = (settings.L, settings.L)

    # Launch kernel to populate the initial values and wait for completion
    AMDGPU.wait(AMDGPU.@roc groupsize=threads gridsize=grid populate!(u,
                                                                      v,
                                                                      roc_offsets,
                                                                      roc_sizes,
                                                                      minL,
                                                                      maxL))

    # Get the faces needed for MPI communication between processes
    xy_face_t, xz_face_t, yz_face_t = Simulation.get_MPI_faces(size_x, size_y, size_z, T)

    # Return the Fields structure containing u, v, and their temporary arrays
    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

"""
ROCm kernel function that populates the initial values for the chemical concentrations
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
    lz = (AMDGPU.workgroupIdx().x - Int32(1)) * AMDGPU.workgroupDim().x +
         AMDGPU.workitemIdx().x
    ly = (AMDGPU.workgroupIdx().y - Int32(1)) * AMDGPU.workgroupDim().y +
         AMDGPU.workitemIdx().y

    # Boundary check for array dimensions
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
using AMD GPU acceleration.

# Arguments
- `fields::Fields`: The fields containing the concentrations of the chemical substances
- `settings::Settings`: Simulation settings containing parameters like diffusion rates
- `mcd::MPICartDomain`: The MPI Cartesian domain information

# No return value (operates in-place)
"""
function Simulation.calculate!(::Val{:amdgpu}, ::Val{:plain},
                               fields::Fields{T, N, <:ROCArray{T, N}},
                               settings::Settings,
                               mcd::MPICartDomain) where {T, N}
    # Convert simulation parameters to the specified type
    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    # Transfer process sizes to GPU memory
    roc_sizes = ROCArray(mcd.proc_sizes)

    # Configure ROCm kernel execution parameters
    threads = (16, 16)
    blocks = (settings.L, settings.L)

    # Launch kernel to perform calculations and wait for completion
    AMDGPU.wait(AMDGPU.@roc groupsize=threads gridsize=grid calculate!(fields.u,
                                                                       fields.v,
                                                                       fields.u_temp,
                                                                       fields.v_temp,
                                                                       roc_sizes,
                                                                       Du,
                                                                       Dv,
                                                                       F,
                                                                       K,
                                                                       noise,
                                                                       dt))
end

"""
ROCm kernel function that performs the actual calculation of the Gray-Scott
equations for each grid point.

# Arguments
- `u`, `v`: Current state of the chemical concentrations
- `u_temp`, `v_temp`: Temporary arrays for storing the next state
- `sizes`: Sizes of the local subdomain
- `Du`, `Dv`: Diffusion coefficients for chemicals U and V
- `F`, `K`: Feed and kill rates for the Gray-Scott model
- `noise`: Magnitude of random fluctuations
- `dt`: Time step size

# No return value (operates in-place)
"""
function calculate!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                    noise, dt)
    # Calculate local coordinates (1-indexed)
    k = (AMDGPU.workgroupIdx().x - Int32(1)) * AMDGPU.workgroupDim().x +
        AMDGPU.workitemIdx().x
    j = (AMDGPU.workgroupIdx().y - Int32(1)) * AMDGPU.workgroupDim().y +
        AMDGPU.workitemIdx().y

    # Process only non-ghost cells
    if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
        # Bounds are inclusive
        for i in 2:(sizes[1] + 1)
            u_ijk = u[i, j, k]
            v_ijk = v[i, j, k]

            # Calculate the gradient of `u` and `v` using the Laplacian operator
            # Note: Random noise term is currently disabled pending AMDGPU.jl support
            du = Du * Simulation.laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                 F * (1.0 - u_ijk)
            # + noise * AMDGPU.rand(eltype(u))
            # WIP in AMDGPU.jl, works with CUDA.jl
            # + rand(Distributions.Uniform(-1, 1))

            dv = Dv * Simulation.laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
                 (F + K) * v_ijk

            # Write back the new values to grid point (i, j, k)
            u_temp[i, j, k] = u_ijk + du * dt
            v_temp[i, j, k] = v_ijk + dv * dt
        end
    end
end

"""
Returns a copy of the fields `u` and `v` without the ghost cells, transferred from GPU to CPU memory.

# Arguments
- `fields::Fields`: The fields containing the concentrations of `u` and `v`

# Returns
- `u_no_ghost::Array{T, N}`: The field `u` without ghost cells
- `v_no_ghost::Array{T, N}`: The field `v` without ghost cells
"""
function Simulation.get_fields(::Val{:amdgpu}, ::Val{:plain},
                               fields::Fields{T, N, <:ROCArray{T, N}}) where {T, N}
    # Transfer data from GPU to CPU and remove ghost cells
    u = Array(fields.u)
    u_no_ghost = u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]

    v = Array(fields.v)
    v_no_ghost = v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]
    return u_no_ghost, v_no_ghost
end

end # module AMDGPUExt
