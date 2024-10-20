"""
Submodule used by GrayScott to handle inputs
"""
module Inputs

export get_settings

import ArgParse
import TOML

# import directly from parent module (GrayScott)
import ..Settings, ..SettingsKeys

# public facing function
function get_settings(args::Vector{String}, comm)::Settings
    config_file = _parse_args(args)

    # check format extension
    if !endswith(config_file, ".toml")
        ext = split(config_file, ".")[end]
        throw(ArgumentError("config file must be in TOML format. Extension not recognized: $ext\n"))
    end

    config_file_contents = String(read(open(config_file, "r")))

    return _parse_settings_toml(config_file_contents)
end

# local scope functions
function _parse_args(args::Vector{String};
                     error_handler = ArgParse.default_handler)::String
    s = ArgParse.ArgParseSettings(description = "gray-scott workflow simulation example configuration file, Julia version, GrayScott.jl",
                                  exc_handler = error_handler)

    #  @add_arg_table! s begin
    #       "--opt1"               # an option (will take an argument)
    #       "--opt2", "-o"         # another option, with short form
    #       "arg1"                 # a positional argument
    #   end

    ArgParse.@add_arg_table! s begin
        "config_file"
        help = "configuration file"
        arg_type = String
        required = true
    end

    # parse_args return a dictionary with key/value for arguments
    parsed_arguments = ArgParse.parse_args(args, s)

    # key is mandatory, so it's safe to retrieve
    config_file::String = parsed_arguments["config_file"]

    return config_file
end

function _parse_settings_toml(toml_contents::String)::Settings
    config_dict = TOML.parse(toml_contents)
    settings = Settings()

    # Iterate through dictionary pairs
    for (key, value) in config_dict
        # Iterate through predefined keys, else ignore (no error if unrecognized)
        if key in SettingsKeys
            setproperty!(settings, Symbol(key), value)
        end
    end

    return settings
end

end # module
