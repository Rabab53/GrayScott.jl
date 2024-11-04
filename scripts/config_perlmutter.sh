#!/bin/bash

# Using the home directory for now
GS_DIR=$SCRATCH/GrayScott.jl
# remove existing generated Manifest.toml
rm -f $GS_DIR/Manifest.toml
rm -f $GS_DIR/LocalPreferences.toml


module load julia/1.10


# Instantiate the project by installing packages in Project.toml
julia --project=$GS_DIR -e 'using Pkg; Pkg.add("CUDA")'
julia --project=$GS_DIR -e 'using Pkg; Pkg.instantiate()'
