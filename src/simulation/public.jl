# Public Simulation API

function load_backend_and_lang(settings::Settings)
    # "cpu", "cuda", "amdgpu"
    lowercase_backend = lowercase(settings.backend)
    backend_symbol = Symbol(lowercase_backend)

    # "plain", "kernelabstractions"
    lowercase_lang = lowercase(settings.kernel_language)
    lang_symbol = Symbol(lowercase_lang)

    return backend_symbol, lang_symbol
end

"""
Create and Initialize fields for either CPU, CUDA.jl, AMDGPU.jl backends,
and plain or KernelAbstractions kernel languages.

Relies on multiple dispatch to call the correct implementation, via a
combination of backend and kernel language.
"""
function init_fields(settings::Settings,
                     mcd::MPICartDomain, T)::Fields{T}
    backend_symbol, lang_symbol = load_backend_and_lang(settings)
    return init_fields(Val{backend_symbol}(), Val{lang_symbol}(), settings, mcd, T)
end

"""
Fallback definition, in case the selected backend/kernel language combination
is not supported.
"""
init_fields(::Val{backend}, ::Val{lang}, settings, mcd, T) where {backend, lang} =
    error("Backend :$backend and kernel language :$lang not supported.")

function iterate!(fields::Fields, settings::Settings, mcd::MPICartDomain)
    backend_symbol, lang_symbol = load_backend_and_lang(settings)
    return iterate!(Val{backend_symbol}(), Val{lang_symbol}(), fields, settings, mcd)
end

iterate!(::Val{backend}, ::Val{lang}, fields, settings, mcd) where {backend, lang} =
    error("Backend :$backend and kernel language :$lang not supported.")
