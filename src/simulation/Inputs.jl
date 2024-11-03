module Inputs

export get_settings

import ArgParse
import TOML

import ..Settings, ..SettingsKeys

"""
Reads the configuration file (a TOML-formatted file), specified in `args`, and
returns a `Settings` object containing the configuration parameters.

# Arguments
- `args::Vector{String}`: Command line arguments

# Returns
- `Settings`: A `Settings` object containing the configuration parameters
"""
function get_settings(args::Vector{String})::Settings
    # Parse command line arguments to get the configuration file path
    config_file = parse_args(args)

    # Validate that the file is in TOML format
    if !endswith(config_file, ".toml")
        ext = split(config_file, ".")[end]
        throw(ArgumentError("Config file must be in TOML format. Extension not recognized: $ext\n"))
    end

    # Read the contents of the configuration file as a String
    config_file_contents = String(read(open(config_file, "r")))

    # Parse the TOML contents and return a Settings object
    return parse_settings_toml(config_file_contents)
end

"""
Parses the command line arguments to get the configuration file path.

# Arguments
- `args::Vector{String}`: Command line arguments
- `error_handler::Function`: ArgParse-compatible error handler function

# Returns
- `String`: The path to the configuration file
"""
function parse_args(args::Vector{String};
                    error_handler = ArgParse.default_handler)
    # Define the settings for the ArgParse parser
    s = ArgParse.ArgParseSettings(description = "gray-scott workflow simulation example configuration file, Julia version, GrayScott.jl",
                                  exc_handler = error_handler)

    # Add the configuration file argument
    ArgParse.@add_arg_table! s begin
        "config_file"
        help = "configuration file"
        arg_type = String
        required = true
    end

    # Parse arguments into a dictionary
    parsed_arguments = ArgParse.parse_args(args, s)

    # Retrieve the configuration file path
    config_file::String = parsed_arguments["config_file"]

    return config_file
end

"""
Parses the TOML-formatted contents of the configuration file and returns a
`Settings` object containing the configuration parameters.

# Arguments
- `toml_contents::String`: The contents of the configuration file in TOML format

# Returns
- `Settings`: A `Settings` object containing the configuration parameters
"""
function parse_settings_toml(toml_contents::String)
    # Parse the TOML contents into a dictionary
    config_dict = TOML.parse(toml_contents)

    # Create a new Settings object
    settings = Settings()

    # Iterate through key/value pairs
    for (key, value) in config_dict
        # Iterate through predefined keys, else ignore (no error if key is unrecognized)
        if key in SettingsKeys
            # Set the value of the key in the Settings object
            setproperty!(settings, Symbol(key), value)
        end
    end

    return settings
end

"""
Given a `Settings` object, returns the backend and kernel language symbols,
which when wrapped in `Val`, can be used to dispatch to the correct backend and
kernel language implementation of various simulation-related functions.

# Arguments
- `settings::Settings`: A `Settings` object containing the configuration parameters

# Returns
- `Tuple{Symbol, Symbol}`: A tuple containing the backend and kernel language symbols
"""
function load_backend_and_lang(settings::Settings)
    # One of "cpu", "cuda", "amdgpu"
    lowercase_backend = lowercase(settings.backend)
    backend_symbol = Symbol(lowercase_backend)

    # One of "plain", "kernelabstractions"
    lowercase_lang = lowercase(settings.kernel_language)
    lang_symbol = Symbol(lowercase_lang)

    return backend_symbol, lang_symbol
end

end # module
