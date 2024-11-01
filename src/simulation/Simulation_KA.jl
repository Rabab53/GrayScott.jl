import KernelAbstractions
import KernelAbstractions: @kernel, @index

KA_backend(::Val{:cpu}) = KernelAbstractions.CPU()
KA_backend(::Val{backend_symbol}) where {backend_symbol} =
    error("Unknown backend: $backend_symbol")

# This is necessary since KernelAbstrations doesn't provide an
# allocate-and-copy function
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
- `Fields`: The fields for the simulation
"""
function init_fields(::Val{backend_symbol}, ::Val{:kernelabstractions},
                     settings::Settings,
                     mcd::MPICartDomain,
                     T)::Fields{T, 3} where {backend_symbol}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    Backend = KA_backend(Val{backend_symbol}())

    u = KernelAbstractions.ones(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)

    u_temp = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)

    cu_offsets = similar_KA(Backend, mcd.proc_offsets)
    cu_sizes = similar_KA(Backend, mcd.proc_sizes)

    d::Int64 = 6
    minL = Int32(settings.L / 2 - d)
    maxL = Int32(settings.L / 2 + d)

    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])

    kernel! = populate_kernel!(Backend, threads)
    kernel!(u, v,
            cu_offsets,
            cu_sizes,
            minL, maxL;
            ndrange=nrange)

    KernelAbstractions.synchronize(Backend)
    xy_face_t, xz_face_t, yz_face_t = get_MPI_faces(size_x, size_y, size_z, T)

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

@kernel function populate_kernel!(u, v, offsets, sizes, minL, maxL)
    # local coordinates (this are 1-index already)
    #lz = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x +
         #CUDA.threadIdx().x
    #ly = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y +
         #CUDA.threadIdx().y

    lz, ly = @index(Global, NTuple)

    if lz <= size(u, 3) && ly <= size(u, 2)
        # get global coordinates
        z = lz + offsets[3] - 1
        y = ly + offsets[2] - 1

        if z >= minL && z <= maxL && y >= minL && y <= maxL
            xoff = offsets[1]

            for x in minL:maxL
                # check if global coordinates for initialization are inside the region
                if !is_inside(x, y, z, offsets, sizes)
                    continue
                end

                # Julia is 1-index, like Fortran :)
                u[x - xoff + 2, ly + 1, lz + 1] = 0.25
                v[x - xoff + 2, ly + 1, lz + 1] = 0.33
            end
        end
    end
end

"""
Calculates the Gray-Scott reaction-diffusion equations for the fields `u` and `v`
for the current process's local subdomain.

# Arguments:
- `fields::Fields`: The fields containing the concentrations of the chemical substances.
- `settings::Settings`: Simulation settings containing parameters like diffusion rates.

# No return value (operates in-place)
"""
function calculate!(::Val{backend_symbol}, ::Val{:kernelabstractions},
                    fields::Fields{T, N},
                    settings::Settings,
                    mcd::MPICartDomain) where {backend_symbol, T, N}
    @kernel function calculate_kernel!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                                       noise, dt)
        k, j = @index(Global, NTuple)

        # loop through non-ghost cells
        if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
            # bounds are inclusive
            for i in 2:(sizes[1] + 1)
                u_ijk = u[i, j, k]
                v_ijk = v[i, j, k]

                du = Du * laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                     F * (1.0 - u_ijk) +
                     noise * rand(Distributions.Uniform(-1, 1))

                dv = Dv * laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
                     (F + K) * v_ijk

                # advance the next step
                u_temp[i, j, k] = u_ijk + du * dt
                v_temp[i, j, k] = v_ijk + dv * dt
            end
        end
    end

    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    Backend = KA_backend(Val{backend_symbol}())

    cu_sizes = similar_KA(Backend, mcd.proc_sizes)

    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])

    kernel! = calculate_kernel!(Backend, threads)
    kernel!(fields.u,
            fields.v,
            fields.u_temp,
            fields.v_temp,
            cu_sizes,
            Du, Dv, F, K,
            noise, dt,
            ndrange=nrange)

    KernelAbstractions.synchronize(Backend)
end
