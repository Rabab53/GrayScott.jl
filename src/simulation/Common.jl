"""
Performs a 7-point Laplacian stencil calculation for a given grid point (i, j, k).

# Arguments:
- `i::Int`: The x-coordinate of the grid point.
- `j::Int`: The y-coordinate of the grid point.
- `k::Int`: The z-coordinate of the grid point.
- `var::Array{T, N}`: The 3D array containing the field values.

# Returns:
- `l::T`: The result of the Laplacian stencil calculation at the grid point.
"""
function laplacian(i, j, k, var)
    @inbounds l = var[i - 1, j, k] + var[i + 1, j, k] + var[i, j - 1, k] +
                  var[i, j + 1, k] + var[i, j, k - 1] + var[i, j, k + 1] -
                  6.0 * var[i, j, k]
    return l / 6.0
end

"""
Define a helper function to check if a given (x, y, z) point is inside the
local domain for this process (not in ghost cells).

# Arguments:
- `x::Int`: The x-coordinate of the point.
- `y::Int`: The y-coordinate of the point.
- `z::Int`: The z-coordinate of the point.
- `offsets::Tuple{Int, Int, Int}`: The offsets of the local domain.
- `sizes::Tuple{Int, Int, Int}`: The sizes of the local domain.

# Returns:
- `Bool`: True if the point is inside the local domain, false otherwise.
"""
function is_inside(x, y, z, offsets, sizes)
    # Check whether the point is within the local domain
    if x < offsets[1] || x >= offsets[1] + sizes[1]
        return false
    end
    if y < offsets[2] || y >= offsets[2] + sizes[2]
        return false
    end
    if z < offsets[3] || z >= offsets[3] + sizes[3]
        return false
    end

    return true
end
