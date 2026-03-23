#!/bin/bash
#------------------------------------------------------------
# Compiles FTorch
#------------------------------------------------------------

cd FTorch

# Exit on error
set -e

# ----- Build type
# Set to Debug for bounds-checking and -O0, Release for -O3
# BUILD_TYPE=Release
BUILD_TYPE=Debug

# ----- Set compiler
FORT_DIR=/opt/local/EOS/nadc_extern/bullseye/gcc-12
export FC=${FORT_DIR}/bin/gfortran-12
export CXX=${FORT_DIR}/bin/g++-12
export CC=${FORT_DIR}/bin/gcc-12

# Build and install FTorch locally
FTORCH_SRC="FTorch"
FTORCH_BUILD="build_ftorch"
FTORCH_INSTALL="FTorch_INSTALLED"
LIBTORCH_DIR="LibTorch/libtorch/"

cmake -S $FTORCH_SRC -B $FTORCH_BUILD \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/$FTORCH_INSTALL \
      -DCMAKE_Fortran_COMPILER=$FC \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_PREFIX_PATH=$(pwd)/$LIBTORCH_DIR \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE
cmake --build $FTORCH_BUILD -j
cmake --install $FTORCH_BUILD

cd ..
echo "FTorch build complete. Installed in FTorch/FTorch_INSTALLED"