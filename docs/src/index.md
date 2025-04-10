# Capse.jl

`Mapse.jl` is a Julia package designed to emulate the computation of the Linear and Nonlinear Matter Power Spectrum, with a speedup of several orders of magnitude compared to standard codes such as `CAMB` or `CLASS`. The core functionalities of `Mapse.jl` are inherithed by the upstream library [`AbstractCosmologicalEmulators.jl`](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl).

## Installation

In order to install  `Mapse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Mapse.jl")
```

## Usage

In order to be able to use `Mapse.jl`, there are two major steps that need to be performed:

- Instantiating the emulators, e.g. initializing the Neural Network, its weights and biases, and the quantities employed in pre and post-processing
- Use the instantiated emulators to retrieve the spectra

In the reminder of this section we are showing how to do this.

### Instantiation

The most direct way to instantiate an official trained emulators is given by the following one-liner

```julia
Pk_emu = Mapse.load_emulator(weights_folder);
```

where `weights_folder` is the path to the folder containing the files required to build up the network.

It is possible to pass an additional argument to the previous function, which is used to choose between the two NN backend now available:

- [SimpleChains](https://github.com/PumasAI/SimpleChains.jl), which is taylored for small NN running on a CPU
- [Lux](https://github.com/LuxDL/Lux.jl), which can run both on CPUs and GPUs

`SimpleChains.jl` is faster expecially for small NNs on the CPU. If you wanna use something running on a GPU, you should use `Lux.jl`, which can be loaded adding an additional argument to the `load_emulator` function, `Capse.LuxEmulator`

```julia
Pk_emu = Capse.load_emulator(weights_folder, emu = Mapse.LuxEmulator);
```

Each trained emulator should be shipped with a description within the JSON file. In order to print the description, just run:

```@example tutorial
Mapse.get_emulator_description(Pk_emu)
```

!!! warning

    Cosmological parameters must be fed to ` Mapse.jl` with **arrays**. It is the user
    responsability to check the right ordering, by reading the output of the
    `get_emulator_description` method.

After loading a trained emulator, feed it some input parameters `x` in order to get the
emulated $P(k,z)$'s

```julia
x = rand(6) # generate some random input
Mapse.get_Pk(x, Pk_emu) #compute the Pk's
```

`SimpleChains.jl` and `Lux.jl` have almost the same performance and they give the same result up to floating point precision.

These benchmarks have been performed locally, with a 13th Gen Intel® Core™ i7-13700H, using a single core.

Considering that a high-precision settings calculation performed with [`CAMB`](https://github.com/cmbant/CAMB) on the same machine requires around 60 seconds, `Mapse.jl` is 5-6 order of magnitudes faster.

!!! warning

    Currently, there is a performance issue when using `Lux.jl` in a multi-threaded scenario. This is
    something known (see discussion [here](https://github.com/LuxDL/Lux.jl/issues/847)).
    In case you want to launch multiple chains locally, the suggested (working) strategy with `Lux.jl`
    is to use distributed computing.

### Authors

- Marco Bonici, PostDoctoral researcher at Waterloo Center for Astrophysics

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you
would like to change.

Please make sure to update tests as appropriate.

### License

`Mapse.jl` is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Mapse.jl/blob/main/LICENSE) for the full
license text.
