# OPERA-S5 NN forward model

This is a standalone Fortran implementation of the S5 neural network (NN) forward models developed within the OPERA-S5 project. The implemented subroutines are meant to be integrated into existing retrieval codes, replacing more expensive physical forward models. Note that the NN forward models emulate the entire forward model chain, including both radiative transfer and convolution with the instrument response function. A separate NN is used for each spectral window (here NIR, SWIR1, and SWIR3).

## Installation

### Cloning the repository

To get started, clone this repository by running
```
git clone git@github.com:FionaLippert/Fortran_NN_forward_model.git
```
to clone via SSH, or
```
git clone https://github.com/FionaLippert/Fortran_NN_forward_model.git
```
to clone via HTTPS.

Now, switch to the project directory:
```
cd Fortran_NN_forward_model
```

### Dependencies

This NN forward model implementation has the following dependencies:
- [FTorch](https://github.com/Cambridge-ICCS/FTorch) (v1.0.0)
- HDF5, NetCDF (C), and NetCDF (Fortran) libraries

To install FTorch, first create the `FTorch` directory (if it does not exist yet):
```
mkdir -p FTorch
```

Then, install LibTorch (C++ version of PyTorch) by running
```
cd FTorch
mkdir -p LibTorch
cd LibTorch
wget "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip"
unzip libtorch-shared-with-deps-2.8.0+cpu.zip
cd ../..
```
Make sure to get the appropriate link for your OS from [PyTorch.org](https://pytorch.org/get-started/locally/#start-locally).
Note that, for now, we only use CPU computations. But in principle, LibTorch and FTorch can be setup to use the GPU if available.

Then, clone the FTorch GitHub repository by running
```
cd FTorch
git clone git@github.com:Cambridge-ICCS/FTorch.git
cd ..
```
to clone via SSH, or 
```
cd FTorch
git clone https://github.com/Cambridge-ICCS/FTorch.git
cd ..
```
to clone via HTTPS.

Finally, install FTorch by running
```
bash compile_ftorch.sh
```
Make sure to adjust the C, C++, and Fortran compiler paths in `compile_ftorch.sh`. For any FTorch-related questions or issues, see the FTorch [README](https://github.com/Cambridge-ICCS/FTorch) and [documentation](https://cambridge-iccs.github.io/FTorch/page/cmake.html).


### Module installation

Now, you can install the S5 NN forward model module by running
```
bash compile.sh
```
Make sure to adjust the path to the Fortran compiler in `compile.sh`, and to point CMake to the correct HDF5 and NetCDF installation in `CMakeLists.txt`.


## Getting started

After compilation, you can execute a simple example program by running
```
./build_nn_fwd/s5_nn_forward_model
```
This includes the following steps:
1. Allocate memory for inputs, outputs, and intermediate data structures.
2. Load the trained NNs and other static information, such as information on spectral windows and ISRF, from disk (all relevant data is included in the `data` folder).
3. Define example inputs to run the forward model for.
4. Use the NN forward model to compute radiances in the NIR, SWIR1, and SWIR3 bands of S5. Here, we do this in two different ways: either by running the NN forward model for each spectral window separately, or calling a pixel-level subroutine for a combined array of wavelengths, which will automatically find the corresponding spectral windows and NN models for each wavelength of interest.
5. Compare the output of these two different approaches, making sure that they do the same thing.
6. Deallocate memory for all data structures allocated in step 1.

## Code structure

The NN forward model implementation is split into four main parts:
1. `main_<approach>.f90`: A small example program running a single iteration of the forward model using dummy inputs (see previous section).
2. `<approach>_nn_data_type_module.f90`: Defines a data container for all relevant inputs, outputs, and intermediate data structured needed when running the NN forward model. This includes custom allocation and deallocation subroutines.
3. `<approach>_nn_forward_model.f90`: Implements the actual NN forward model, including the two main entry points `forward_model_pixel_nn_<approach>` and `forward_model_win_nn_<approach>` for pixel-level and spectral window-level forward models respectively.
4. `<approach>_nn_forward_model_jacobian.f90`: Additional functions for Jacobian computations using a combination of finite differences and analytical derivatives. This may or may not be needed, depending on the retrieval code.

This repository includes the implementations of two different approaches: `base` and `hybrid`:
- The `base` NN forward model directly predicts radiances for each spectral window from the provided inputs.
- The `hybrid` NN forward model relies on a non-scattering forward model to obtain approximate radiances for each spectral window, which are then corrected by the NN components. Note that the `hybrid` code can also be used to run the `base` model by setting `USE_NS_INPUT = .FALSE.` in `hybrid_nn_forward_model.f90`.

By default the code is compiled using the hybrid approach. Switching between approaches can be achieved by simply adjusting the source files in `CMakeLists.txt`
