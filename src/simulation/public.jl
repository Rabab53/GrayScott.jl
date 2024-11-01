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
    backend_symbol, lang_symbol = load_backend_and_lang(settings)

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
Runs the simulation for a single time step.

# Arguments:
- `fields::Fields`: The fields for the simulation.
- `settings::Settings`: Simulation settings containing parameters like grid size.
- `mcd::MPICartDomain`: The local subdomain configuration for the current process.

# No return value (operates in-place).
"""
function iterate!(fields::Fields, settings::Settings, mcd::MPICartDomain)
    # Extract the backend and kernel language symbols from the settings
    backend_symbol, lang_symbol = load_backend_and_lang(settings)

    # Delegate to the appropriate backend-specific function
    iterate!(Val{backend_symbol}(), Val{lang_symbol}(), fields, settings, mcd)

    return
end

"""
Fallback definition, in case the selected backend/kernel language combination
is not supported.
"""
iterate!(::Val{backend}, ::Val{lang}, fields, settings, mcd) where {backend, lang} =
    error("Backend :$backend and kernel language :$lang not supported.")
