PROGRAM main_hybrid
    USE hybrid_nn_forward_model_module
    USE hybrid_nn_data_type_module
    USE ftorch
    IMPLICIT NONE

    !--- Constants
    INTEGER, PARAMETER :: N_WIN = 3                                  ! number of spectral windows (here: NIR, SWIR1, SWIR3)
    INTEGER, PARAMETER :: N_IN(N_WIN) = [9, 12, 11]                  ! number of NN inputs per window
    INTEGER, PARAMETER :: N_LAYERS = 19                              ! number of pressure layers in atmospheric profiles (number of levels - 1)
    INTEGER, PARAMETER :: N_IN_PROFILES(N_WIN) = [1, 1, 1]           ! number of profile variables (here: temperature only) per window
    INTEGER, PARAMETER :: N_GASES(N_WIN) = [0, 3, 2]                 ! number of gas inputs per window
    INTEGER, PARAMETER :: N_WAVE_MEAS(N_WIN) = [211, 851, 801]       ! number of measurement wavelengths per window
    INTEGER, PARAMETER :: N_WAVE_FINE(N_WIN) = [62069, 33963, 20239] ! number of fine wavelengths per window
    INTEGER, PARAMETER :: N_ILS_ISRF(N_WIN) = [13793, 1887, 1191]    ! number of ISRF points per window
    CHARACTER(LEN=200), PARAMETER :: DATA_PATH = '/deos/fional/gitlab/s5_nn_forward_model/data/'
    ! CHARACTER(LEN=200), PARAMETER :: NN_NAME = 'traced_base_emulator_OPERA-S5'
    CHARACTER(LEN=200), PARAMETER :: NN_NAME = 'traced_hybrid_emulator_OPERA-S5'

    !--- Variable declarations

    ! internal arrays & data structures
    TYPE(hybrid_nn_data_type), DIMENSION(N_WIN) :: hybrid_nn_data_array
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
    DOUBLE PRECISION :: p_surf
    DOUBLE PRECISION :: albedo
    DOUBLE PRECISION :: size_param
    DOUBLE PRECISION :: aod_550
    DOUBLE PRECISION :: aer_height
    DOUBLE PRECISION :: h2o_col
    DOUBLE PRECISION :: co2_col
    DOUBLE PRECISION :: ch4_col

    ! temperature profile array
    DOUBLE PRECISION, DIMENSION(N_LAYERS) :: temp_profile


    !----- Allocate memory

    ! Allocate data for all spectral windows
    DO iwin = 1, N_WIN
        CALL allocate_hybrid_nn_data(hybrid_nn_data_array(iwin), &
                N_IN(iwin), N_LAYERS, N_IN_PROFILES(iwin), N_GASES(iwin), &
                N_WAVE_MEAS(iwin), N_WAVE_FINE(iwin), N_ILS_ISRF(iwin))
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
    CALL load_all_win_data(win_data_path, hybrid_nn_data_array)
    DO iwin = 1, N_WIN
        ! Load neural network model for this spectral window
        CALL load_win_hybrid_nn(nn_path, iwin, hybrid_nn_data_array(iwin)%nn_model)
    END DO


    !----- Prepare inputs

    ! Example inputs
    cos_sza = 0.49112011871274269 !1.0D0         ! cosine of solar zenith angle
    cos_vza = 0.88246296874272301 !1.0D0         ! cosine of viewing zenith angle
    rel_az = -1.2988297889021516 !1.0D0          ! relative azimuth angle (radians)
    p_surf = 900.0                             ! surface pressure (hPa)
    albedo = 8.1776843569132757E-002 !0.2D0          ! avg albedo in spectral window
    size_param = 4.0D0                               ! size parameter
    ! aod_550 = 2436968717013.3159           ! total column number
    aod_550 = 0.01D0         ! AOD at 550nm
    aer_height = 2180.8362054323525 !500.0D0    ! ALH (m)
    h2o_col = 4.7087457089815220E+022 ! 4.69D22       ! H2O column (molecules/m^2)
    co2_col = 9.1473413145705631E+021 ! 9.11D21       ! CO2 column (molecules/m^2)
    ch4_col = 4.0141328593958806E+019 ! 4.05D19       ! CH4 column (molecules/m^2)

    temp_profile = [223.7755, 211.3333, 209.1118, 209.0873, 213.8197, 221.2127, 230.7730, 239.4532, 247.1599, 253.3649, &
                    258.7442, 263.6776, 268.0436, 271.8163, 275.4351, 278.8109, 281.8099, 284.7354, 287.3052]


    ! compute scattering angle (radians) from geometry
    cos_scat = -cos_sza * cos_vza + &
            SQRT((1.D0 - cos_sza**2)) * SQRT((1.D0 - cos_vza**2)) * &
            COS(rel_az)
    scat_angle = ACOS(cos_scat)

    write(*,*) "Scattering angle (rad): ", scat_angle

    ! Combine NN inputs into array, for all spectral windows
    DO iwin = 1, N_WIN
        CALL prepare_win_nn_inputs(iwin, cos_sza, cos_vza, rel_az, scat_angle, p_surf, &
                albedo, size_param, aod_550, aer_height, h2o_col, co2_col, ch4_col, &
                hybrid_nn_data_array(iwin)%nn_inputs)

        write(*,*) "NN inputs for window ", iwin, ": ", hybrid_nn_data_array(iwin)%nn_inputs

        ! set profile inputs
        hybrid_nn_data_array(iwin)%nn_profile_inputs(:, 1) = temp_profile
        
        ! for now, set dummy values for fine-res inputs
        hybrid_nn_data_array(iwin)%tau_abs_fine(:) = 4.0D0
        hybrid_nn_data_array(iwin)%xbdrf_wave_fine(:) = albedo
        
        ! Initialize other variables for non-scattering calculation
        hybrid_nn_data_array(iwin)%cos_sza = cos_sza
        hybrid_nn_data_array(iwin)%cos_vza = cos_vza

    END DO

    !----- Make predictions for all spectral windows

    ! Run Hybrid NN forward model (incl. non-scattering approximation)
    ! for each spectral window separately
    DO iwin = 1, N_WIN
        write(*,*) "Running NN forward model for window ", iwin
        CALL forward_model_win_hybrid_nn(hybrid_nn_data_array(iwin))
        write(*,*) "Predicted spectrum min/max for window ", iwin, ": ", &
            MINVAL(hybrid_nn_data_array(iwin)%spectrum), MAXVAL(hybrid_nn_data_array(iwin)%spectrum)
    END DO

    ! Now, directly run the full pixel-level function
    ! First, prepare combined wavelength array
    j = 1
    DO iwin = 1, N_WIN
        wl(j:j+N_WAVE_MEAS(iwin)-1) = hybrid_nn_data_array(iwin)%wl_meas
        j = j + N_WAVE_MEAS(iwin)
    END DO
    ! Now, run everything at once (matching wavelengths to spectral windows internally)
    write(*,*) "Running pixel-level NN forward model for entire wavelength range"
    CALL forward_model_pixel_hybrid_nn(wl, N_WIN, hybrid_nn_data_array, radiances)

    ! Assert that results are the same as when running window-by-window
    j = 1
    DO iwin = 1, N_WIN
        IF (MAXVAL(ABS(radiances(j:j+N_WAVE_MEAS(iwin)-1) - &
                       hybrid_nn_data_array(iwin)%spectrum)) > 1.D-8) THEN
            PRINT *, "Error: results are not matching for iwin = ", iwin
            write(*, *) " --- min/max error for iwin = ", iwin, ": ", &
                MINVAL(ABS(radiances(j:j+N_WAVE_MEAS(iwin)-1) - hybrid_nn_data_array(iwin)%spectrum)), &
                MAXVAL(ABS(radiances(j:j+N_WAVE_MEAS(iwin)-1) - hybrid_nn_data_array(iwin)%spectrum))
        ELSE
            PRINT *, "Success: results are matching for iwin = ", iwin
        END IF
        j = j + N_WAVE_MEAS(iwin)
    END DO

    !----- Cleanup

    DO iwin = 1, N_WIN
        CALL deallocate_hybrid_nn_data(hybrid_nn_data_array(iwin))
    END DO

    DEALLOCATE(wl)
    DEALLOCATE(radiances)

END PROGRAM main_hybrid