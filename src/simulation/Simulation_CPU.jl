function Init_Fields_CPU(settings::Settings,
                          mcd::MPICartDomain, T)::Fields{T}
    size_x = mcd.proc_sizes[1]
    size_y = mcd.proc_sizes[2]
    size_z = mcd.proc_sizes[3]

    # should be ones
    u = ones(T, size_x + 2, size_y + 2, size_z + 2)
    v = zeros(T, size_x + 2, size_y + 2, size_z + 2)

    u_temp = zeros(T, size_x + 2, size_y + 2, size_z + 2)
    v_temp = zeros(T, size_x + 2, size_y + 2, size_z + 2)

    d::Int64 = 6

    # global locations
    minL = Int64(settings.L / 2 - d)
    maxL = Int64(settings.L / 2 + d)

    xoff = mcd.proc_offsets[1]
    yoff = mcd.proc_offsets[2]
    zoff = mcd.proc_offsets[3]

    Threads.@threads for z in minL:maxL
        for y in minL:maxL
            for x in minL:maxL
                if !Is_Inside(x, y, z, mcd.proc_offsets, mcd.proc_sizes)
                    continue
                end

                # Julia is 1-index, like Fortran :)
                u[x - xoff + 2, y - yoff + 2, z - zoff + 2] = 0.25
                v[x - xoff + 2, y - yoff + 2, z - zoff + 2] = 0.33
            end
        end
    end

    xy_face_t, xz_face_t, yz_face_t = Get_MPI_Faces(size_x, size_y, size_z, T)

    fields = Fields(u, v, u_temp, v_temp, xy_face_t, xz_face_t, yz_face_t)
    return fields
end

function Iterate!(fields::Fields{T, N, Array{T, N}}, settings::Settings,
                  mcd::MPICartDomain) where {T, N}
    Exchange!(fields, mcd)
    # this function is the bottleneck
    Calculate!(fields, settings, mcd)

    # swap the names
    fields.u, fields.u_temp = fields.u_temp, fields.u
    fields.v, fields.v_temp = fields.v_temp, fields.v
end

function Calculate!(fields::Fields{T, N, Array{T, N}}, settings::Settings,
                     mcd::MPICartDomain) where {T, N}
    Du = convert(T, settings.Du)
    Dv = convert(T, settings.Dv)
    F = convert(T, settings.F)
    K = convert(T, settings.k)
    noise = convert(T, settings.noise)
    dt = convert(T, settings.dt)

    # loop through non-ghost cells, bounds are inclusive
    # @TODO: load balancing? option: a big linear loop
    # use @inbounds at the right for-loop level, avoid putting it at the top level
    Threads.@threads for k in 2:(mcd.proc_sizes[3] + 1)
        for j in 2:(mcd.proc_sizes[2] + 1)
            @inbounds for i in 2:(mcd.proc_sizes[1] + 1)
                u = fields.u[i, j, k]
                v = fields.v[i, j, k]

                # introduce a random disturbance on du
                du = Du * Laplacian(i, j, k, fields.u) - u * v^2 +
                     F * (1.0 - u) +
                     noise * rand(Distributions.Uniform(-1, 1))

                dv = Dv * Laplacian(i, j, k, fields.v) + u * v^2 -
                     (F + K) * v

                # advance the next step
                fields.u_temp[i, j, k] = u + du * dt
                fields.v_temp[i, j, k] = v + dv * dt
            end
        end
    end
end

function Get_Fields(fields::Fields{T, N, Array{T, N}}) where {T, N}
    @inbounds begin
        u_no_ghost = fields.u[(begin + 1):(end - 1), (begin + 1):(end - 1),
                              (begin + 1):(end - 1)]
        v_no_ghost = fields.v[(begin + 1):(end - 1), (begin + 1):(end - 1),
                              (begin + 1):(end - 1)]
    end
    return u_no_ghost, v_no_ghost
end