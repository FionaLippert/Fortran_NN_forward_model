!------------------------------------------------------------------------------
!  MODULE mod_s5nn_derived_type
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the derived types used in the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE mod_s5nn_derived_type
    USE ftorch
    IMPLICIT NONE

    !--- Constants
    ! number of spectral windows (here: NIR, SWIR1, SWIR3)
    INTEGER, PARAMETER :: N_WIN = 3
    ! number of NN inputs per window
    INTEGER, PARAMETER :: N_IN(N_WIN) = [8, 11, 10]
    ! number of measurement wavelengths per window
    INTEGER, PARAMETER :: N_WAVE_MEAS(N_WIN) = [211, 851, 801]
    ! total number of measurement wavelengths (all windows concatenated)
    INTEGER, PARAMETER :: TOTAL_NWAVE_MEAS = SUM(N_WAVE_MEAS)
    ! number of fine wavelengths per window
    INTEGER, PARAMETER :: N_WAVE_FINE(N_WIN) = [62069, 33963, 20239]
    ! total number of fine wavelengths (all windows concatenated)
    INTEGER, PARAMETER :: TOTAL_NWAVE_FINE = SUM(N_WAVE_FINE)
    ! min and max indices on fine wavelength grid per window
    INTEGER, PARAMETER :: MIN_IDX_WAVE_FINE(N_WIN) = [1, 62070, 96033]
    INTEGER, PARAMETER :: MAX_IDX_WAVE_FINE(N_WIN) = [62069, 96032, 116271]
    ! number of ISRF points per window
    INTEGER, PARAMETER :: N_ILS_ISRF(N_WIN) = [13793, 1887, 1191]
    ! path to the data directory
    CHARACTER(LEN=256), PARAMETER :: DATA_PATH = '../data/'
    ! name of the neural network model (and corresponding directory in data/nn_models/)
    CHARACTER(LEN=256), PARAMETER :: NN_NAME = 'traced_ratio_mlp_spectrometer_only_fixed_log_albedo_log_reff_1e6_dropout0001'


    !--- Public subroutines implemented in this module
    PUBLIC :: define_save_state_s5nn

    !--- Data types defined in this module
    ! GRASP data structures
    PUBLIC :: forward_model_characteristics_s5nn
    PUBLIC :: forward_model_save_s5nn
    PUBLIC :: forward_model_configuration_s5nn
    ! Internal data structures
    PUBLIC :: s5nn_win_data
    PUBLIC :: s5nn_data

    ! Everything else in this module is private
    PRIVATE

    !--- Data types (for GRASP interface)

    ! Characteristics related to the core S5 NN forward model implementation
    ! (derived from RIN settings, internal data, and output data)
    ! this should contain data that may change during the retrieval 
    ! (i.e. is updated based on state vector, during unpacking)
    TYPE forward_model_characteristics_s5nn

        !--- Satellite geometry
        DOUBLE PRECISION :: cos_sza
        DOUBLE PRECISION :: cos_vza
        DOUBLE PRECISION :: rel_az
        DOUBLE PRECISION :: scat_angle

        !--- Surface parameters
        DOUBLE PRECISION, DIMENSION(N_WIN) :: avg_albedo

        !--- Aerosol parameters
        DOUBLE PRECISION :: angstrom_exponent
        DOUBLE PRECISION :: aod_550
        DOUBLE PRECISION :: aer_height

        !--- Gas columns
        DOUBLE PRECISION :: h2o_column
        DOUBLE PRECISION :: co2_column
        DOUBLE PRECISION :: ch4_column

        !--- Surface and gas data on fine wavelength grid
        ! surface BRDF on fine wavelength grid (all windows concatenated)
        DOUBLE PRECISION, DIMENSION(TOTAL_NWAVE_FINE) :: xbdrf_wave_fine
        ! absorption optical depth on fine wavelength grid (all windows concatenated)
        DOUBLE PRECISION, DIMENSION(TOTAL_NWAVE_FINE) :: tau_abs_fine

        !--- Output data
        ! radiances on measurement wavelength grid (all windows concatenated)
        DOUBLE PRECISION, DIMENSION(TOTAL_NWAVE_MEAS) :: radiances

    END TYPE forward_model_characteristics_s5nn

    ! Configuration parameters related to the core S5 NN forward model implementation
    ! (logic flags)
    TYPE forward_model_configuration_s5nn

        ! Flag to indicate whether to use the hybrid NN or the base NN forward model
        LOGICAL :: use_hybrid_nn
        
        ! Flag to indicate whether to match wavelength channels directly or interpolate
        LOGICAL :: interpolate

    END TYPE forward_model_configuration_s5nn

    !---- Data types (internal)
    ! Internal data structure for each spectral window
    TYPE s5nn_win_data

        ! the traced PyTorch neural network, loaded via FTorch
        TYPE(torch_model) :: nn_model

        ! inputs and outputs for the neural network
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: nn_inputs            ! input array for the neural network (n_nn_inputs)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_spectrum          ! non-scattering spectrum array (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: spectrum             ! final spectrum (n_wave_meas)
        TYPE(torch_tensor), DIMENSION(1) :: input_tensors                   ! first input tensor holding input array (n_nn_inputs)
        TYPE(torch_tensor), DIMENSION(1) :: proxy_tensors                   ! second input tensor holding non-scattering array (n_wave_meas)
        TYPE(torch_tensor), DIMENSION(1) :: output_tensors                  ! output tensor holding final spectrum (n_wave_meas)

        ! data needed for measurement channel matching
        DOUBLE PRECISION, ALLOCATABLE :: wl_meas(:)                         ! measurement wavelengths in nm (n_wave_meas)
        DOUBLE PRECISION :: res_meas                                        ! measurement resolution in nm (assumed constant)
        INTEGER :: n_wave_meas                                              ! number of measurement wavelengths

        ! data needed for or filled by non-scattering calculations
        DOUBLE PRECISION :: cos_sza                                         ! cosine of solar zenith angle
        DOUBLE PRECISION :: cos_vza                                         ! cosine of viewing zenith angle
        DOUBLE PRECISION, ALLOCATABLE :: wl_fine(:)                         ! fine wavelength grid in nm (n_wave_fine)
        DOUBLE PRECISION, ALLOCATABLE :: resp_isrf(:, :)                    ! ISRF response function (n_ils_isrf, n_wave_meas)
        DOUBLE PRECISION, ALLOCATABLE :: dw_isrf(:, :)                      ! ISRF delta wavelength grid (n_ils_isrf, n_wave_meas)
        DOUBLE PRECISION, ALLOCATABLE :: solar_irrad_fine(:)                ! solar irradiance (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: xbdrf_wave_fine      ! surface BRDF on fine grid (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: tau_abs_fine         ! absorption optical depth on fine grid (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_spectrum_fine     ! non-scattering spectrum on fine grid (n_wave_fine)
        INTEGER :: n_ils_isrf                                               ! number of ISRF points
        INTEGER :: n_wave_fine                                              ! number of fine wavelengths
        DOUBLE PRECISION :: res_fine                                        ! fine wavelength grid resolution (nm)

    END TYPE s5nn_win_data

    ! Internal data structure holding data for all spectral windows
    TYPE s5nn_data
        CHARACTER(LEN=256) :: nn_path
        CHARACTER(LEN=256) :: win_data_path
        TYPE(s5nn_win_data), DIMENSION(N_WIN) :: win_data
    END TYPE s5nn_data

    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Defines which variables to save for later computations
    SUBROUTINE define_save_state_s5nn()
        IMPLICIT NONE
        !------------------------------------------------------------------------------

        ! TODO: specify which variables to save for later computations

    END SUBROUTINE define_save_state_s5nn


END MODULE mod_s5nn_derived_type