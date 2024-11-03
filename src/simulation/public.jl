# Public Simulation API

"""
Creates and initializes the fields (u and v) for the simulation on either CPU, CUDA.jl, or
AMDGPU.jl backends, and using either plain Julia or KernelAbstractions kernel languages.

This function delegates to the appropriate backend-specific function using
multiple dispatch on the configured backend and kernel language.

# Arguments:
- `settings::Settings`: Simulation settings containing parameters like grid size.
- `mcd::MPICartDomain`: The local subdomain configuration for the current process.
- `T`: Element type for the fields (e.g., `Float64`).

# Returns:
- `Fields{T}`: The initialized fields for the simulation.
"""
function init_fields(settings::Settings,
                     mcd::MPICartDomain, T)::Fields{T}
    # Extract the backend and kernel language symbols from the settings
    backend_symbol, lang_symbol = Inputs.load_backend_and_lang(settings)

    # Delegate to the appropriate backend-specific function
    return init_fields(Val{backend_symbol}(), Val{lang_symbol}(), settings, mcd, T)
end

"""
Fallback definition, in case the selected backend/kernel language combination
is not supported.
"""
init_fields(::Val{backend}, ::Val{lang}, settings, mcd, T) where {backend, lang} =
    error("Backend :$backend and kernel language :$lang not supported.")

"""
Iterates over the fields for a single time step, updating the concentrations of the
chemical substances `u` and `v` based on the Gray-Scott reaction-diffusion equations.

# Arguments
- `fields::Fields`: The fields containing the concentrations of the chemical substances.
- `settings::Settings`: Simulation settings containing parameters like diffusion rates.
- `mcd::MPICartDomain`: The local subdomain configuration for the current process.

# No return value (operates in-place)
"""
function iterate!(fields::Fields, settings::Settings, mcd::MPICartDomain)
    # Extract the backend and kernel language symbols from the settings
    backend_symbol, lang_symbol = Inputs.load_backend_and_lang(settings)

    # Delegate to the appropriate backend-specific function
    iterate!(Val{backend_symbol}(), Val{lang_symbol}(), fields, settings, mcd)

    return
end
function iterate!(::Val{backend_symbol}, ::Val{lang_symbol},
                  fields::Fields{T, N, Array{T, N}},
                  settings::Settings,
                  mcd::MPICartDomain) where {backend_symbol, lang_symbol, T, N}
    # Perform the exchange of ghost cells between neighboring processes
    # This function is communication-bound
    exchange!(fields, mcd)

    # Calculate the new values for the fields `u` and `v` based on the Gray-Scott equations
    # This function is compute/memory-bound
    calculate!(Val{backend_symbol}(), Val{lang_symbol}(), fields, settings, mcd)

    # Swap the fields to prepare for the next iteration
    fields.u, fields.u_temp = fields.u_temp, fields.u
    fields.v, fields.v_temp = fields.v_temp, fields.v

    return
end

"""
Fallback definition, in case the selected backend/kernel language combination
is not supported.
"""
iterate!(::Val{backend}, ::Val{lang}, fields, settings, mcd) where {backend, lang} =
    error("Backend :$backend and kernel language :$lang not supported.")
