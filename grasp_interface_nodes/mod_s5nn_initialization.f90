!------------------------------------------------------------------------------
!  MODULE mod_s5nn_initialization
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the initialization routines for the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE mod_s5nn_initialization
    USE mod_s5nn_derived_type, ONLY: N_WIN, DATA_PATH, NN_NAME, s5nn_win_data, s5nn_data
    USE netcdf
    USE ftorch
    IMPLICIT NONE


    !--- Public subroutines implemented in this module
    PUBLIC :: initialize_module_s5nn
    PUBLIC :: initialize_forward_model_characteristics_s5nn
    PUBLIC :: initialize_forward_model_config_s5nn
    PUBLIC :: initialize_s5nn_data

    ! Everything else in this module is private
    PRIVATE

    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Initializes the S5 neural network forward model module
    !------------------------------------------------------------------------------
    SUBROUTINE initialize_module_s5nn()
        IMPLICIT NONE

        ! TODO: any module-level initialization?

    END SUBROUTINE initialize_module_s5nn

    !------------------------------------------------------------------------------
    !> @brief Initializes the forward_model_characteristics_s5nn data type
    !> @param[inout] forw_s5nn The data structure to initialize
    !------------------------------------------------------------------------------
    SUBROUTINE initialize_forward_model_characteristics_s5nn(forw_s5nn)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        TYPE(forward_model_characteristics_s5nn), INTENT(INOUT) :: forw_s5nn
        !--------------------------------------------------------------------------

        ! Initialize NN inputs
        forw_s5nn%cos_sza = 0.0D0
        forw_s5nn%cos_vza = 0.0D0
        forw_s5nn%rel_az = 0.0D0
        forw_s5nn%cos_scat = 0.0D0
        forw_s5nn%scat_angle = 0.0D0
        forw_s5nn%avg_albedo(:) = 0.0D0 ! TODO: remove this and derive from xbdrf_wave_fine?
        forw_s5nn%aod_550 = 0.0D0
        forw_s5nn%angstrom = 0.0D0
        forw_s5nn%aer_height = 0.0D0
        forw_s5nn%h2o_col = 0.0D0
        forw_s5nn%co2_col = 0.0D0
        forw_s5nn%ch4_col = 0.0D0

        ! Initialize additional data needed for non-scattering calculations
        forw_s5nn%xbdrf_wave_fine(:) = 0.0D0
        forw_s5nn%tau_abs_fine(:) = 0.0D0
        ! Initialize output data
        forw_s5nn%radiances(:) = 0.0D0

    END SUBROUTINE initialize_forward_model_characteristics_s5nn

    !------------------------------------------------------------------------------
    !> @brief Initializes the forward_model_configuration_s5nn data type
    !> @param[inout] fw_cfg_s5nn The data structure to initialize
    !------------------------------------------------------------------------------
    SUBROUTINE initialize_forward_model_config_s5nn(fw_cfg_s5nn)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        TYPE(forward_model_configuration_s5nn), INTENT(INOUT) :: fw_cfg_s5nn
        !--------------------------------------------------------------------------

        ! Initialize configuration parameters
        fw_cfg_s5nn%use_hybrid_nn = .FALSE.
        fw_cfg_s5nn%interpolate = .FALSE.

    END SUBROUTINE initialize_forward_model_config_s5nn

    !------------------------------------------------------------------------------
    !> @brief Initializes the internal s5nn_data structure
    !> @param[inout] nn_data The s5nn_data object to initialize
    !------------------------------------------------------------------------------
    SUBROUTINE initialize_s5nn_data(nn_data)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        TYPE(s5nn_data), INTENT(INOUT) :: nn_data
        !--------------------------------------------------------------------------
        ! Initialize NN input and output arrays
        ! Initialize paths
        nn_data%nn_path = TRIM(DATA_PATH) // "nn_models/" // TRIM(NN_NAME) // "/"
        nn_data%win_data_path = TRIM(DATA_PATH) // "win_data/"

        ! Initialize the neural network data for each spectral window
        DO iwin = 1, N_WIN
            ! Load static information about this spectral window
            CALL load_win_data(nn_data%win_data_path, iwin, nn_data%win_data(iwin))
            ! Load neural network model for this spectral window
            CALL load_win_nn(nn_data%nn_path, iwin, nn_data%win_data(iwin)%nn_model)
        END DO
    END SUBROUTINE initialize_s5nn_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_win_data
    !> @brief Load static data for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] win_data_path Path to directory containing window NetCDF files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object containing all necessary data
    !------------------------------------------------------------------------------
    SUBROUTINE load_win_data(win_data_path, iwin, win_data)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: win_data_path
        INTEGER, INTENT(IN) :: iwin
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data
        !--------------------------------------------------------------------------
        ! Local variables for netCDF reading
        INTEGER :: ncid, status
        INTEGER :: dimid_nwave_meas, dimid_nwave_fine, dimid_nils_isrf
        INTEGER :: nwave_meas, nwave_fine, nils_isrf
        CHARACTER(LEN=50) :: nwave_meas_name, nwave_fine_name, nils_isrf_name
        INTEGER :: varid_dw, varid_resp, varid_wl_meas, varid_wl_fine, varid_solar_irrad
        !--------------------------------------------------------------------------
        CHARACTER(LEN=10) :: win_str

        IF (iwin == 1) THEN
            win_str = 'iwin_1'
        ELSE IF (iwin == 2) THEN
            win_str = 'iwin_2'
        ELSE IF (iwin == 3) THEN
            win_str = 'iwin_3'
        ELSE
            PRINT *, "Error: Invalid window index ", iwin
            STOP 1
        END IF

        ! --- Open netCDF file
        status = nf90_open(TRIM(win_data_path) // 'win_data_' // TRIM(win_str) // '.nc', NF90_NOWRITE, ncid)
        CALL handle_nf90_error(status, "Could not open netCDF file.")

        ! --- Get dimensions
        status = nf90_inq_dimid(ncid, "nils_isrf", dimid_nils_isrf)
        CALL handle_nf90_error(status, "Could not find dimension 'nils_isrf' in file.")
        status = nf90_inquire_dimension(ncid, dimid_nils_isrf, nils_isrf_name, nils_isrf)
        CALL handle_nf90_error(status, "Could not inquire dimension 'nils_isrf'.")

        status = nf90_inq_dimid(ncid, "nwave_meas", dimid_nwave_meas)
        CALL handle_nf90_error(status, "Could not find dimension 'nwave_meas' in file.")
        status = nf90_inquire_dimension(ncid, dimid_nwave_meas, nwave_meas_name,nwave_meas)
        CALL handle_nf90_error(status, "Could not inquire dimension 'nwave_meas'.")

        status = nf90_inq_dimid(ncid, "nwave_fine", dimid_nwave_fine)
        CALL handle_nf90_error(status, "Could not find dimension 'nwave_fine' in file.")
        status = nf90_inquire_dimension(ncid, dimid_nwave_fine, nwave_fine_name, nwave_fine)
        CALL handle_nf90_error(status, "Could not inquire dimension 'nwave_fine'.")

        ! --- Get variables
        status = nf90_inq_varid(ncid, "wl_meas", varid_wl_meas)
        CALL handle_nf90_error(status, "Could not find variable 'wl_meas' in file.")
        status = nf90_get_var(ncid, varid_wl_meas, win_data%wl_meas)
        CALL handle_nf90_error(status, "Could not read variable 'wl_meas' from file.")

        status = nf90_inq_varid(ncid, "wl_fine", varid_wl_fine)
        CALL handle_nf90_error(status, "Could not find variable 'wl_fine' in file.")
        status = nf90_get_var(ncid, varid_wl_fine, win_data%wl_fine)
        CALL handle_nf90_error(status, "Could not read variable 'wl_fine' from file.")

        status = nf90_inq_varid(ncid, "solar_irradiance", varid_solar_irrad)
        CALL handle_nf90_error(status, "Could not find variable 'solar_irradiance' in file.")
        status = nf90_get_var(ncid, varid_solar_irrad, win_data%solar_irrad_fine)
        CALL handle_nf90_error(status, "Could not read variable 'solar_irradiance' from file.")

        status = nf90_inq_varid(ncid, "dw_isrf", varid_dw)
        CALL handle_nf90_error(status, "Could not find variable 'dw_isrf' in file.")
        status = nf90_get_var(ncid, varid_dw, win_data%dw_isrf)
        CALL handle_nf90_error(status, "Could not read variable 'dw_isrf' from file.")

        status = nf90_inq_varid(ncid, "resp_isrf", varid_resp)
        CALL handle_nf90_error(status, "Could not find variable 'resp_isrf' in file.")
        status = nf90_get_var(ncid, varid_resp, win_data%resp_isrf)
        CALL handle_nf90_error(status, "Could not read variable 'resp_isrf' from file.")

        ! --- Close
        status = nf90_close(ncid)
        CALL handle_nf90_error(status, "Could not close netCDF file.")

        ! --- Store dimensions and other relevant properties
        win_data%res_meas = &
                win_data%wl_meas(2) - &
                win_data%wl_meas(1)
        win_data%res_fine = &
                win_data%wl_fine(2) - &
                win_data%wl_fine(1)
        win_data%n_wave_meas = SIZE(win_data%wl_meas)
        win_data%n_wave_fine = SIZE(win_data%wl_fine)
        win_data%n_ils_isrf = SIZE(win_data%dw_isrf, 1)

    END SUBROUTINE load_win_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_win_nn
    !> @brief Load neural network model for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] nn_path Path to directory containing NN related files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[out] nn_win Loaded neural network model for the specified window
    !------------------------------------------------------------------------------
    SUBROUTINE load_win_nn(nn_path, iwin, nn_win)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: nn_path
        INTEGER, INTENT(IN) :: iwin
        TYPE(torch_model), INTENT(OUT) :: nn_win
        !--------------------------------------------------------------------------
        ! Local variables
        CHARACTER(LEN=10) :: win_str
        !--------------------------------------------------------------------------
        IF (iwin == 1) THEN
            win_str = 'iwin_1'
        ELSE IF (iwin == 2) THEN
            win_str = 'iwin_2'
        ELSE IF (iwin == 3) THEN
            win_str = 'iwin_3'
        ELSE
            PRINT *, "Error: Invalid window index ", iwin
            STOP 1
        END IF

        ! Load neural network model for the specified spectral window
        CALL torch_model_load(nn_win, TRIM(nn_path) // TRIM(win_str) // '.pt', torch_kCPU)

    END SUBROUTINE load_win_nn

    !------------------------------------------------------------------------------
    ! SUBROUTINE handle_nf90_error
    !> @brief Handle netCDF error by printing message and stopping execution
    !> @param[in] status netCDF status code
    !> @param[in] msg Error message to display
    SUBROUTINE handle_nf90_error(status, msg)
        IMPLICIT NONE
        !---------------------------------------------------------------------------
        INTEGER, INTENT(IN) :: status
        CHARACTER(LEN=*), INTENT(IN) :: msg
        !---------------------------------------------------------------------------
        ! Local variables
        CHARACTER(LEN=*) :: tmp_message
        !--------------------------------------------------------------------------
        IF (status /= 0) THEN
            ! Throw GRASP error
            write(tmp_message,'(a)') &
                    "NetCDF error: ", TRIM(nf90_strerror(status)), &
                     " Status code: ", status, &
                     " Message: ", TRIM(msg)
            G_ERROR(trim(tmp_message))
        END IF

    END SUBROUTINE handle_nf90_error

END MODULE mod_s5nn_initialization