#!/bin/bash
#------------------------------------------------------------
# Compiles the S5 NN forward model project
#------------------------------------------------------------

# Exit on error
set -e

# ----- Build type
# Set to Debug for bounds-checking and -O0, Release for -O3
# BUILD_TYPE=Release
BUILD_TYPE=Debug

# --- Optional OpenMP ---
# ENABLE_OMP=yes

# ----- Set compiler
FORT_DIR=/opt/local/EOS/nadc_extern/bullseye/gcc-12
export FC=${FORT_DIR}/bin/gfortran-12

# Build and install the S5 NN forward model
mkdir -p build_nn_fwd
cd build_nn_fwd

cmake .. \
    -DCMAKE_Fortran_COMPILER=$FC \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_PREFIX_PATH="$FTORCH_INSTALL;$FORT_DIR"

cmake --build . -j

cd ..
echo "Build complete. Executable is in build_nn_fwd/s5_nn_forward_model"
