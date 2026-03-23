PROGRAM main_base
    USE base_nn_forward_model_module
    USE base_nn_data_type_module
    USE ftorch
    IMPLICIT NONE

    !--- Constants
    INTEGER, PARAMETER :: N_WIN = 3                                  ! number of spectral windows (here: NIR, SWIR1, SWIR3)
    INTEGER, PARAMETER :: N_IN(N_WIN) = [8, 11, 10]                  ! number of NN inputs per window
    INTEGER, PARAMETER :: N_WAVE_MEAS(N_WIN) = [211, 851, 801]       ! number of measurement wavelengths per window
    CHARACTER(LEN=200), PARAMETER :: DATA_PATH = '/deos/fional/gitlab/s5_nn_forward_model/data/'
    CHARACTER(LEN=200), PARAMETER :: NN_NAME = 'traced_simple_mlp_spectrometer_only_fixed_log_albedo_log_reff_1e6_dropout0001'

    !--- Variable declarations

    ! internal arrays & data structures
    TYPE(base_nn_data_type), DIMENSION(N_WIN) :: base_nn_data_array
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: wl
    DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: radiances

    ! data paths
    CHARACTER(LEN=200) :: win_data_path
    CHARACTER(LEN=200) :: nn_model_path
    CHARACTER(LEN=200) :: nn_path

    INTEGER :: iwin
    INTEGER :: j

    DOUBLE PRECISION :: cos_sza
    DOUBLE PRECISION :: cos_vza
    DOUBLE PRECISION :: rel_az
    DOUBLE PRECISION :: cos_scat
    DOUBLE PRECISION :: scat_angle
    DOUBLE PRECISION :: albedo
    DOUBLE PRECISION :: aod_550
    DOUBLE PRECISION :: angstrom
    DOUBLE PRECISION :: aer_height
    DOUBLE PRECISION :: h2o_col
    DOUBLE PRECISION :: co2_col
    DOUBLE PRECISION :: ch4_col


    !----- Allocate memory

    ! Allocate data for all spectral windows
    DO iwin = 1, N_WIN
        CALL allocate_base_nn_data(base_nn_data_array(iwin), &
                N_IN(iwin), N_WAVE_MEAS(iwin))
    END DO

    ! Allocate arrays for full pixel-level function
    ALLOCATE(wl(SUM(N_WAVE_MEAS)))
    ALLOCATE(radiances(SUM(N_WAVE_MEAS)))

    !----- Setup NN model and static inputs

    ! Define path to window data and nn models
    win_data_path = TRIM(DATA_PATH) // "win_data/"
    nn_model_path = TRIM(DATA_PATH) // "nn_models/"
    nn_path = TRIM(nn_model_path) // TRIM(NN_NAME) // "/"

    ! Load static properties of each spectral window,
    ! incl. wavelength grids, ISRF, solar irradiance
    CALL load_all_win_data(win_data_path, base_nn_data_array)
    DO iwin = 1, N_WIN
        ! Load neural network model for this spectral window
        CALL load_win_nn(nn_path, iwin, base_nn_data_array(iwin)%nn_model)
    END DO


    !----- Prepare inputs

    ! Example inputs
    cos_sza = 1.0D0         ! cosine of solar zenith angle
    cos_vza = 1.0D0         ! cosine of viewing zenith angle
    rel_az = 1.0D0          ! relative azimuth angle (radians)
    albedo = 0.2D0          ! avg albedo in spectral window
    angstrom = 3.0D0        ! angstrom exponent
    aod_550 = 0.1D0         ! AOD at 550nm
    aer_height = 500.0D0    ! ALH (m)
    h2o_col = 4.69D22       ! H2O column (molecules/m^2)
    co2_col = 9.11D21       ! CO2 column (molecules/m^2)
    ch4_col = 4.05D19       ! CH4 column (molecules/m^2)

    ! compute scattering angle (radians) from geometry
    cos_scat = -cos_sza * cos_vza + &
            SQRT((1.D0 - cos_sza**2)) * SQRT((1.D0 - cos_vza**2)) * &
            COS(rel_az)
    scat_angle = ACOS(cos_scat)

    ! Combine NN inputs into array, for all spectral windows
    DO iwin = 1, N_WIN
        CALL prepare_win_nn_inputs(iwin, cos_sza, cos_vza, rel_az, scat_angle, &
                albedo, angstrom, aod_550, aer_height, h2o_col, co2_col, ch4_col, &
                base_nn_data_array(iwin)%nn_inputs)
    END DO

    !----- Make predictions for all spectral windows

    ! Run NN forward model (incl. non-scattering approximation)
    ! for each spectral window separately
    DO iwin = 1, N_WIN
        write(*,*) "Running NN forward model for window ", iwin
        CALL forward_model_win_nn(base_nn_data_array(iwin))
    END DO

    ! Now, directly run the full pixel-level function
    ! First, prepare combined wavelength array
    j = 1
    DO iwin = 1, N_WIN
        wl(j:j+N_WAVE_MEAS(iwin)-1) = base_nn_data_array(iwin)%wl_meas
        j = j + N_WAVE_MEAS(iwin)
    END DO
    ! Now, run everything at once (matching wavelengths to spectral windows internally)
    write(*,*) "Running pixel-level NN forward model for entire wavelength range"
    CALL forward_model_pixel_nn(wl, N_WIN, base_nn_data_array, radiances)

    ! Assert that results are the same as when running window-by-window
    j = 1
    DO iwin = 1, N_WIN
        IF (MAXVAL(ABS(radiances(j:j+N_WAVE_MEAS(iwin)-1) - &
                       base_nn_data_array(iwin)%spectrum)) > 1.D-8) THEN
            PRINT *, "Error: results are not matching for iwin = ", iwin
        ELSE
            PRINT *, "Success: results are matching for iwin = ", iwin
        END IF
        j = j + N_WAVE_MEAS(iwin)
    END DO

    !----- Cleanup

    DO iwin = 1, N_WIN
        CALL deallocate_base_nn_data(base_nn_data_array(iwin))
    END DO

    DEALLOCATE(wl)
    DEALLOCATE(radiances)

END PROGRAM main_base