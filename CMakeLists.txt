cmake_minimum_required(VERSION 3.12)
project(s5_nn_forward_model LANGUAGES Fortran)

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -Wall -Wextra -fcheck=all -fbacktrace -fbounds-check
                        -ffree-line-length-none -ffixed-line-length-none
                        -ffpe-trap=invalid,zero,overflow,underflow
                        -fallow-argument-mismatch)
else()
    add_compile_options(-g -O3 -ffree-line-length-none
                        # -ffpe-trap=invalid,zero,overflow,underflow
                        -fallow-argument-mismatch)
endif()

# Real*8 support (equivalent to compile_real8=yes)
add_compile_options(-fdefault-real-8 -fdefault-double-8)


# Paths
set(FTORCH_DIR "${CMAKE_SOURCE_DIR}/FTorch/FTorch_INSTALLED")
set(NETCDF_DIR "/opt/local/EOS/nadc_extern/bullseye/gcc-12")  # headers & libraries in include/lib
set(HDF5_DIR "${NETCDF_DIR}")  # same as NetCDF in this setup

# Source files
add_executable(s5_nn_forward_model
    src/main_hybrid.f90
    src/hybrid_nn_forward_model.f90
    src/hybrid_nn_data_type_module.f90
)
# add_executable(s5_nn_forward_model
#     src/main_base.f90
#     src/base_nn_forward_model.f90
#     src/base_nn_data_type_module.f90
# )

# Include directories
target_include_directories(s5_nn_forward_model PRIVATE
    ${FTORCH_DIR}/include/ftorch
    ${NETCDF_DIR}/include
    ${HDF5_DIR}/include
)

# Libraries (manual linking)
target_link_libraries(s5_nn_forward_model PRIVATE
    ${FTORCH_DIR}/lib/libftorch.so
    ${NETCDF_DIR}/lib/libnetcdff.a
    ${NETCDF_DIR}/lib/libnetcdf.a
    ${HDF5_DIR}/lib/libhdf5_fortran.a
    ${HDF5_DIR}/lib/libhdf5_hl.a
    ${HDF5_DIR}/lib/libhdf5.a
    gomp pthread z gfortran dl stdc++
)

# FTorch runtime path
set_target_properties(s5_nn_forward_model PROPERTIES
    INSTALL_RPATH "${FTORCH_DIR}/lib"
)