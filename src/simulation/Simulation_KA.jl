using KernelAbstractions
const DEV = :NVIDIA


if DEV == :NVIDIA
    using CUDA
    const ArrayKA = CUDA.CuArray
    const Backend = CUDABackend()
elseif DEV == :AMD
    using AMDGPU
    const ArrayKA = AMDGPU.ROCArrays
    const Backend = ROCBackend()
elseif DEV == :oneAPI
    using oneAPI 
    const ArrayKA = oneAPI.oneArray
    const Backend = oneAPIBackend()
elseif DEV == :Metal
    using Metal 
    const ArrayKA = Metal.MetalArray
    const Backend = MetalBackend()
else DEV == :CPU
    const ArrayKA = Array
    const Backend = CPU()
end

function Init_Fields_CUDA(settings::Settings, mcd::MPICartDomain,
                           T)::Fields{T, 3, <:ArrayKA{T, 3}}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # should be ones
    #u = CUDA.ones(T, size_x + 2, size_y + 2, size_z + 2)
    #v = CUDA.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    u =  KernelAbstractions.ones(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v =  KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)

    u_temp =  KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    v_temp =  KernelAbstractions.zeros(Backend, T, size_x + 2, size_y + 2, size_z + 2)
    
    cu_offsets = ArrayKA(mcd.proc_offsets)
    cu_sizes = ArrayKA(mcd.proc_sizes)

    d::Int64 = 6
    minL = Int32(settings.L / 2 - d)
    maxL = Int32(settings.L / 2 + d)

    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])

    kernel! = Populate_CUDA_Kernel!(Backend, threads)
    kernel!(u, v, 
            cu_offsets, 
            cu_sizes, 
            minL, maxL, 
            ndrange=nrange)

    KernelAbstractions.synchronize(Backend)
    xy_face_t, xz_face_t, yz_face_t = Get_MPI_Faces(size_x, size_y, size_z, T)

    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

function Iterate!(fields::Fields{T, N, <:ArrayKA{T, N}},
                  settings::Settings,
                  mcd::MPICartDomain) where {T, N}
    Exchange!(fields, mcd)
    # this function is the bottleneck
    Calculate!(fields, settings, mcd)

    # swap the names
    fields.u, fields.u_temp = fields.u_temp, fields.u
    fields.v, fields.v_temp = fields.v_temp, fields.v
end



@kernel function Populate_CUDA_Kernel!(u, v, offsets, sizes, minL, maxL)


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
                # check if global coordinates for Initialization are inside the region
                if !Is_Inside(x, y, z, offsets, sizes)
                    continue
                end

                # Julia is 1-index, like Fortran :)
                u[x - xoff + 2, ly + 1, lz + 1] = 0.25
                v[x - xoff + 2, ly + 1, lz + 1] = 0.33
            end
        end
    end
end

function Calculate!(fields::Fields{T, N, <:ArrayKA{T, N}},
                     settings::Settings,
                     mcd::MPICartDomain) where {T, N}
    @kernel function Calculte_Kernel!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                               noise, dt)

        # local coordinates (this are 1-index already)
        #k = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x +
        #    CUDA.threadIdx().x
        #j = (CUDA.blockIdx().y - Int32(1)) * CUDA.blockDim().y +
        #    CUDA.threadIdx().y

        k, j = @index(Global, NTuple)

        # loop through non-ghost cells
        if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
            # bounds are inclusive
            for i in 2:(sizes[1] + 1)
                u_ijk = u[i, j, k]
                v_ijk = v[i, j, k]

                du = Du * Laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                     F * (1.0 - u_ijk) +
                     noise * rand(Distributions.Uniform(-1, 1))

                dv = Dv * Laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
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

    cu_sizes = ArrayKA(mcd.proc_sizes)

    threads = (16, 16)
    nrange = (settings.L * threads[1], settings.L * threads[2])


    kernel! = Calculte_Kernel!(Backend, threads)
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

function Get_Fields(fields::Fields{T, N, <:ArrayKA{T, N}}) where {T, N}
    u = Array(fields.u)
    u_no_ghost = u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]

    v = Array(fields.v)
    v_no_ghost = v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]
    return u_no_ghost, v_no_ghost
end