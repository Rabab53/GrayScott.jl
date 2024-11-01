module AMDGPUExt

import GrayScott
import GrayScott: Simulation
import GrayScott.Simulation: Settings, Fields, MPICartDomain
import AMDGPU
import AMDGPU: ROCArray

Simulation.KA_backend(::Val{:amdgpu}) = AMDGPU.ROCBackend()

function Simulation.init_fields(::Val{:amdgpu}, ::Val{:plain},
                                settings::Settings,
                                mcd::MPICartDomain,
                                T)::Fields{T, 3, <:ROCArray{T, 3}}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # should be ones
    u = AMDGPU.ones(T, size_x + 2, size_y + 2, size_z + 2)
    v = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    u_temp = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = AMDGPU.zeros(T, size_x + 2, size_y + 2, size_z + 2)

    roc_offsets = ROCArray(mcd.proc_offsets)
    roc_sizes = ROCArray(mcd.proc_sizes)

    d::Int64 = 6
    minL = Int64(settings.L / 2 - d)
    maxL = Int64(settings.L / 2 + d)

    # @TODO: get ideal grid size and threads
    threads = (16, 16)
    # grid size must be the total number of threads of each direction
    grid = (settings.L, settings.L)

    AMDGPU.wait(AMDGPU.@roc groupsize=threads gridsize=grid populate!(u,
                                                                      v,
                                                                      roc_offsets,
                                                                      roc_sizes,
                                                                      minL,
                                                                      maxL))

    xy_face_t, xz_face_t, yz_face_t = Simulation.get_MPI_faces(size_x, size_y, size_z, T)

    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

function populate!(u, v, offsets, sizes, minL, maxL)

    # local coordinates (this are 1-index already)
    lz = (AMDGPU.workgroupIdx().x - Int32(1)) * AMDGPU.workgroupDim().x +
         AMDGPU.workitemIdx().x
    ly = (AMDGPU.workgroupIdx().y - Int32(1)) * AMDGPU.workgroupDim().y +
         AMDGPU.workitemIdx().y

    # This check might not be needed
    if lz <= size(u, 3) && ly <= size(u, 2)

        # get global coordinates
        z = lz + offsets[3] - 1
        y = ly + offsets[2] - 1

        if z >= minL && z <= maxL && y >= minL && y <= maxL
            xoff = offsets[1]

            for x in minL:maxL
                # check if global coordinates for initialization are inside the region
                if !Simulation.is_inside(x, y, z, offsets, sizes)
                    continue
                end

                # Julia is 1-index, like Fortran :)
                u[x - xoff + 2, ly + 1, lz + 1] = 0.25
                v[x - xoff + 2, ly + 1, lz + 1] = 0.33
            end
        end
    end
end

function Simulation.calculate!(::Val{:amdgpu}, ::Val{:plain},
                               fields::Fields{T, N, <:ROCArray{T, N}},
                               settings::Settings,
                               mcd::MPICartDomain) where {T, N}
    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    roc_sizes = ROCArray(mcd.proc_sizes)

    threads = (16, 16)
    blocks = (settings.L, settings.L)

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

function calculate!(u, v, u_temp, v_temp, sizes, Du, Dv, F, K,
                    noise, dt)

    # local coordinates (this are 1-index already)
    k = (AMDGPU.workgroupIdx().x - Int32(1)) * AMDGPU.workgroupDim().x +
        AMDGPU.workitemIdx().x
    j = (AMDGPU.workgroupIdx().y - Int32(1)) * AMDGPU.workgroupDim().y +
        AMDGPU.workitemIdx().y

    # loop through non-ghost cells
    if k >= 2 && k <= sizes[3] + 1 && j >= 2 && j <= sizes[2] + 1
        # bounds are inclusive
        for i in 2:(sizes[1] + 1)
            u_ijk = u[i, j, k]
            v_ijk = v[i, j, k]

            du = Du * Simulation.laplacian(i, j, k, u) - u_ijk * v_ijk^2 +
                 F * (1.0 - u_ijk)
            # + noise * AMDGPU.rand(eltype(u))
            # WIP in AMDGPU.jl, works with CUDA.jl
            # + rand(Distributions.Uniform(-1, 1))

            dv = Dv * Simulation.laplacian(i, j, k, v) + u_ijk * v_ijk^2 -
                 (F + K) * v_ijk

            # advance the next step
            u_temp[i, j, k] = u_ijk + du * dt
            v_temp[i, j, k] = v_ijk + dv * dt
        end
    end
end

function Simulation.get_fields(::Val{:amdgpu}, ::Val{:plain},
                               fields::Fields{T, N, <:ROCArray{T, N}}) where {T, N}
    u = Array(fields.u)
    u_no_ghost = u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]

    v = Array(fields.v)
    v_no_ghost = v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                   (begin + 1):(end - 1)]
    return u_no_ghost, v_no_ghost
end

end # module AMDGPUExt
