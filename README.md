# GrayScott.jl

Julia version of [the gray-scott C++ and Python](https://github.com/ornladios/ADIOS2-Examples/blob/master/source/cpp/gray-scott/) example.

This is a 3D 7-point stencil code to simulate the following [Gray-Scott
reaction diffusion model](https://doi.org/10.1126/science.261.5118.189):

```
u_t = Du * (u_xx + u_yy + u_zz) - u * v^2 + F * (1 - u) + noise * randn(-1,1)
v_t = Dv * (v_xx + v_yy + v_zz) + u * v^2 - (F + k) * v
```

This version contains:

- CPU threaded solver using Julia's [multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/)
- GPU solvers using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl)
- Vendor-agnostic kernel code using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- Parallel I/O using the [ADIOS2.jl](https://github.com/eschnett/ADIOS2.jl) Julia bindings to [ADIOS2](https://github.com/ornladios/ADIOS2)
- Message passing interface (MPI) using [MPI.jl](https://github.com/JuliaParallel/MPI.jl) Julia bindings to MPI
- Easily switch between float- (Float32) and double- (Float64) precision in the configuration file

## How to run a simulation

Pre-requisities:

- A recent Julia version: v1.10 as of May 2023 from [julialang.org/downloads](https://julialang.org/downloads/)

### Run locally

1. **Set up dependencies**

From the `GrayScott.jl` directory instantiate and use MPI artifact jll (preferred method). 
To use a system provided MPI, see [here](https://juliaparallel.org/MPI.jl/latest/configuration/#using_system_mpi)

```julia

$ julia --project

Julia REPL

julia> ]  

(GrayScott.jl)> instantiate
...
(GrayScott.jl)> # press backspace to exit Pkg mode
julia> using MPIPreferences
julia> MPIPreferences.use_jll_binary()
julia> exit()
```

Julia manages its own packages using [Pkg.jl](https://pkgdocs.julialang.org/v1/), the above would create platform-specific `LocalPreferences.toml` and `Manifest.toml` files.

To use a system provided ADIOS2 library, see [here](https://eschnett.github.io/ADIOS2.jl/dev/#Using-a-custom-or-system-provided-ADIOS2-library). 
It just sets the environment variable `JULIA_ADIOS2_PATH` and builds ADIOS2.jl in Julia.

**Note**: GrayScott.jl requires the `main` development version.

    ```
    $ julia
    julia> ]
    GrayScott.jl> add ADIOS2#main  
    ```

2. **Set up the examples/settings-files.toml configuration file**

```
L = 64
Du = 0.2
Dv = 0.1
F = 0.02
k = 0.048
dt = 1.0
plotgap = 10
steps = 1000
noise = 0.1
output = "gs-julia-1MPI-64L-F32.bp"
checkpoint = false
checkpoint_freq = 700
checkpoint_output = "ckpt.bp"
restart = false
restart_input = "ckpt.bp"
adios_config = "adios2.xml"
adios_span = false
adios_memory_selection = false
mesh_type = "image"
precision = "Float32"
backend = "CPU"
```

3. **Running the simulation**

- `CPU threads`: launch julia assigning a number of threads (e.g. -t 8):

    ```
    $ julia --project -t 8 gray-scott.jl examples/settings-files.toml
    ```

- `CUDA/AMDGPU`: set the "backend" option in examples/settings-files.toml to either "CUDA" or "AMDGPU"

    ```
    $ julia --project gray-scott.jl examples/settings-files.toml
    ```

This would generate an adios2 file from the output entry in the configuration file (e.g. `gs-julia-1MPI-64L-F32.bp`)
that can be visualized with ParaView with either the VTX or the FIDES readers.
**Important**: the AMDGPU.jl implementation of `randn` is currently work in progress.
See related issue [here](https://github.com/JuliaGPU/AMDGPU.jl/issues/378)


4. **Running on OLCF Summit and Crusher systems**
The code was tested on the Oak Ridge National Laboratory Leadership Computing Facilities (OLCF): [Summit](https://docs.olcf.ornl.gov/systems/summit_user_guide.html) and [Crusher](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html). Both are used testing a recent version of Julia [v1.9.0-beta3](https://julialang.org/downloads/#upcoming_release) and a `JULIA_DEPOT_PATH` is required to install packages and artifacts. **DO NOT USE your home directory**. We are providing configuration scripts in `scripts/config_XXX.sh` showing the plumming required for these systems. They need to be executed only once per session from the login nodes. 

To reuse these file the first 3 entries must be modified and run on login-nodes and the PATH poiting at a downloaded Julia binary for the corresponding PowerPC (Summit) and x86-64 (Crusher) architectures. Only "CPU" and "CUDA" backends are supported on Summit, while "CPU" and "AMDGPU" backends are supported on Crusher.

    ```
    # Replace these 3 entries
    PROJ_DIR=/gpfs/alpine/proj-shared/csc383
    export JULIA_DEPOT_PATH=$PROJ_DIR/etc/summit/julia_depot
    GS_DIR=$PROJ_DIR/wgodoy/ADIOS2-Examples/source/julia/GrayScott.jl
    ...
    # and the path 
    export PATH=$PROJ_DIR/opt/summit/julia-1.9.0-beta3/bin:$PATH
    ```

## Citation
Please cite the following [paper](https://doi.org/10.1145/3624062.3624278) associated with the code, if you find it useful: 

```
@inproceedings{10.1145/3624062.3624278,
author = {Godoy, William F. and Valero-Lara, Pedro and Anderson, Caira and Lee, Katrina W. and Gainaru, Ana and Ferreira Da Silva, Rafael and Vetter, Jeffrey S.},
title = {Julia as a Unifying End-to-End Workflow Language on the Frontier Exascale System},
year = {2023},
isbn = {9798400707858},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3624062.3624278},
doi = {10.1145/3624062.3624278},
booktitle = {Proceedings of the SC '23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
pages = {1989–1999},
numpages = {11},
keywords = {data analysis, exascale, Frontier supercomputer, Jupyter notebooks, end-to-end workflows, High-Performance Computing, Julia, HPC},
location = {Denver, CO, USA},
series = {SC-W '23}
}
```

## Acknowledgements
This research was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative. This work is funded, in part, by Bluestone, a X-Stack project in the DOE Advanced Scientific Computing Office with program manager Hal Finkel.

This research used resources of the Oak Ridge Leadership Computing Facility and the Experimental Computing Laboratory (ExCL) at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

Thanks to all the Julia community members, packages developers and maintainers for their great work.
