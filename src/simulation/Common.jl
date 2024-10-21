"""
   7-point stencil around the cell, 
   this is equally a host and a device function!
"""
function laplacian(i, j, k, var)
    @inbounds l = var[i - 1, j, k] + var[i + 1, j, k] + var[i, j - 1, k] +
                  var[i, j + 1, k] + var[i, j, k - 1] + var[i, j, k + 1] -
                  6.0 * var[i, j, k]
    return l / 6.0
end


function is_inside(x, y, z, offsets, sizes)::Bool
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
