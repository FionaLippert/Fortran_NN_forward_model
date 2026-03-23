!------------------------------------------------------------------------------
!  MODULE base_nn_forward_model_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the subroutines of the S5 neural network forward
!> model, for the NIR, SWIR1 and SWIR3 bands of the S5/UVNS spectrometer.
!------------------------------------------------------------------------------
MODULE base_nn_forward_model_module
    USE ftorch
    USE netcdf
    USE base_nn_data_type_module
    IMPLICIT NONE
    PRIVATE

    ! Methods implemented in this module
    PUBLIC :: forward_model_pixel_nn
    PUBLIC :: forward_model_win_nn
    PUBLIC :: prepare_win_nn_inputs
    PUBLIC :: load_all_win_data
    PUBLIC :: load_win_data
    PUBLIC :: load_win_nn
    PRIVATE :: handle_nf90_error
    PRIVATE :: interp1d

    ! Predefined flag: Match wavelength channels directly or interpolate?
    ! Interpolation is slower but more flexible
    LOGICAL :: interpolate = .FALSE.

    ! Ftorch internal tensor layouts
    INTEGER, PARAMETER :: input_layout(1) = [1]
    INTEGER, PARAMETER :: output_layout(1) = [1]

    ! Ftorch internal tensor device
    INTEGER, PARAMETER :: torch_device = torch_kCPU

CONTAINS

    
    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_pixel_nn
    !> @brief Forward model for one pixel using neural networks for NIR, SWIR1, SWIR3
    !> @param[in] wl Wavelengths of the measurement channels (n_wave_total)
    !> @param[in] n_win Number of spectral windows
    !> @param[in/out] base_nn_data_array Array of data structures holding data for each spectral window (n_win)
    !> @param[out] radiances Output radiances at measurement wavelengths (n_wave_total)
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_pixel_nn(wl, n_win, base_nn_data_array, radiances)
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: wl
        INTEGER, INTENT(IN) :: n_win
        TYPE(base_nn_data_type), DIMENSION(:), INTENT(INOUT) :: base_nn_data_array
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: radiances
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwave
        INTEGER :: iwin
        INTEGER :: idx
        INTEGER :: n_wave_meas
        LOGICAL :: found_wl
        LOGICAL, DIMENSION(n_win) :: ran_nn_win
        !--------------------------------------------------------------------------

        ! Flags to indicate if NN has been run for a window
        ran_nn_win(:) = .FALSE.

        ! === Loop over spectral channels ===
        DO iwave = 1, SIZE(wl)

            ! Flag to indicate if wavelength was found in any window
            found_wl = .FALSE.

            ! Loop over spectral windows to find which window this wavelength belongs to
            DO iwin = 1, n_win

                ! Get total number of measurement wavelengths for this window
                n_wave_meas = base_nn_data_array(iwin)%n_wave_meas

                IF (wl(iwave) >= base_nn_data_array(iwin)%wl_meas(1) .AND. &
                    wl(iwave) <= base_nn_data_array(iwin)%wl_meas(n_wave_meas)) THEN

                    ! This wavelength belongs to window iwin
                    found_wl = .TRUE.

                    ! If NN has not yet been run for this window, do so now
                    IF (.NOT. ran_nn_win(iwin)) THEN
                        CALL forward_model_win_nn(base_nn_data_array(iwin))
                        ran_nn_win(iwin) = .TRUE.
                    END IF

                    ! Now assign the corresponding output value to the spectrum
                    IF (interpolate) THEN
                        ! Interpolate to measurement wavelength grid
                        radiances(iwave) = interp1d( &
                                base_nn_data_array(iwin)%wl_meas, &
                                base_nn_data_array(iwin)%spectrum, &
                                wl(iwave))
                    ELSE
                        ! Directly match wavelength channels (faster, but requires exact match)
                        idx = INT((wl(iwave) - base_nn_data_array(iwin)%wl_meas(1)) / &
                                               base_nn_data_array(iwin)%res_meas) + 1
                        radiances(iwave) = base_nn_data_array(iwin)%spectrum(idx)
                    END IF
                END IF
            END DO
            
            ! If wavelength was not found in any window, set to zero and show warning
            IF (.NOT. found_wl) THEN
                PRINT *, "Warning: wavelength ", wl(iwave), " outside NN forward model range."
                radiances(iwave) = 0.0d0
            END IF

        END DO

    END SUBROUTINE forward_model_pixel_nn

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_nn
    !> @brief Forward model for one spectral window
    !> @param[in/out] base_nn_data The base_nn_data_type object holding data for the spectral window 
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_nn(base_nn_data)
        ! Input & output parameters
        TYPE(base_nn_data_type), INTENT(INOUT) :: base_nn_data
        !--------------------------------------------------------------------------

        ! Link Fortran arrays to FTorch tensors
        CALL link_tensors_to_arrays(base_nn_data)

        ! Run NN for this window --> fills spectrum array in base_nn_data
        CALL torch_model_forward( &
                base_nn_data%nn_model, &
                [base_nn_data%input_tensors], &
                base_nn_data%output_tensors)

    END SUBROUTINE forward_model_win_nn


    !------------------------------------------------------------------------------
    ! SUBROUTINE prepare_win_nn_inputs
    !> @brief Prepare NN input tensor for a single spectral window
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[in] cos_sza Cosine of solar zenith angle
    !> @param[in] cos_vza Cosine of viewing zenith angle
    !> @param[in] rel_az Relative azimuth angle in radians
    !> @param[in] scat_angle Scattering angle in radians
    !> @param[in] albedo Surface albedo
    !> @param[in] angstrom Angstrom exponent
    !> @param[in] aod_550 Aerosol optical depth (AOD) at 550 nm
    !> @param[in] aer_height Aerosol layer height in m
    !> @param[in] h2o_col Water vapor column in molecules/m^2
    !> @param[in] co2_col CO2 column in molecules/m^2
    !> @param[in] ch4_col CH4 column in molecules/m^2
    !> @param[out] inputs Array of input parameters (n_nn_inputs)
    !------------------------------------------------------------------------------
    SUBROUTINE prepare_win_nn_inputs(iwin, cos_sza, cos_vza, rel_az, scat_angle, albedo, &
                        angstrom, aod_550, aer_height, h2o_col, co2_col, ch4_col, inputs)
        ! Input & output parameters
        INTEGER, INTENT(IN) :: iwin
        DOUBLE PRECISION, INTENT(IN) :: cos_sza
        DOUBLE PRECISION, INTENT(IN) :: cos_vza
        DOUBLE PRECISION, INTENT(IN) :: rel_az
        DOUBLE PRECISION, INTENT(IN) :: scat_angle
        DOUBLE PRECISION, INTENT(IN) :: albedo
        DOUBLE PRECISION, INTENT(IN) :: angstrom
        DOUBLE PRECISION, INTENT(IN) :: aod_550
        DOUBLE PRECISION, INTENT(IN) :: aer_height
        DOUBLE PRECISION, INTENT(IN) :: h2o_col
        DOUBLE PRECISION, INTENT(IN) :: co2_col
        DOUBLE PRECISION, INTENT(IN) :: ch4_col
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: inputs
        !--------------------------------------------------------------------------
        ! Check input array size (different for each window)
        IF (iwin == 1) THEN
            ! assert that inputs has size 8 (0 trace gas inputs for NIR)
            IF(SIZE(inputs) /= 8) THEN
                ! stop and throw error
                PRINT *, "Error: inputs array must have size 8 for NIR window (iwin=1)"
                STOP 1
            END IF
        ELSE IF (iwin == 2) THEN
            ! assert that inputs has size 11 (3 trace gas inputs for SWIR1)
            IF(SIZE(inputs) /= 11) THEN
                ! stop and throw error
                PRINT *, "Error: inputs array must have size 11 for SWIR1 window (iwin=2)"
                STOP 1
            END IF
        ELSE IF (iwin == 3) THEN
            ! assert that inputs has size 10 (2 trace gas inputs for SWIR3)
            IF(SIZE(inputs) /= 10) THEN
                ! stop and throw error
                PRINT *, "Error: inputs array must have size 10 for SWIR3 window (iwin=3)"
                STOP 1
            END IF
        END IF

        ! Fill input array
        inputs(1) = cos_sza
        inputs(2) = cos_vza
        inputs(3) = rel_az
        inputs(4) = scat_angle
        inputs(5) = albedo
        inputs(6) = angstrom
        inputs(7) = aod_550
        inputs(8) = aer_height
        IF (iwin == 2) THEN
            inputs(9) = h2o_col
            inputs(10) = co2_col
            inputs(11) = ch4_col
        ELSE IF (iwin == 3) THEN
            inputs(9) = h2o_col
            inputs(10) = ch4_col
        END IF

    END SUBROUTINE prepare_win_nn_inputs

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_all_win_data
    !> @brief Load relevant data for all spectral windows
    !> @param[in] win_data_path Path to directory containing window data files
    !> @param[in/out] base_nn_data_array Array of base_nn_data_type objects to be filled
    !------------------------------------------------------------------------------
    SUBROUTINE load_all_win_data(win_data_path, base_nn_data_array)
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: win_data_path

        TYPE(base_nn_data_type), DIMENSION(:), INTENT(INOUT) :: base_nn_data_array
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwin
        INTEGER :: n_win
        !--------------------------------------------------------------------------

        ! Determine number of spectral windows
        n_win = SIZE(base_nn_data_array)

        ! Load static data for each spectral window
        DO iwin = 1, n_win
            CALL load_win_data(win_data_path, iwin, base_nn_data_array(iwin))
        END DO

    END SUBROUTINE load_all_win_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_win_data
    !> @brief Load relevant data for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] win_data_path Path to directory containing window data files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[in/out] base_nn_data The base_nn_data_type object to be filled
    !------------------------------------------------------------------------------
    SUBROUTINE load_win_data(win_data_path, iwin, base_nn_data)
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: win_data_path
        INTEGER, INTENT(IN) :: iwin
        TYPE(base_nn_data_type), INTENT(INOUT) :: base_nn_data
        !--------------------------------------------------------------------------
        ! Local variables for netCDF reading
        INTEGER :: ncid, status
        INTEGER :: dimid_nwave_meas
        INTEGER :: nwave_meas
        CHARACTER(LEN=50) :: nwave_meas_name
        INTEGER :: varid_wl_meas
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
        status = nf90_inq_dimid(ncid, "nwave_meas", dimid_nwave_meas)
        CALL handle_nf90_error(status, "Could not find dimension 'nwave_meas' in file.")
        status = nf90_inquire_dimension(ncid, dimid_nwave_meas, nwave_meas_name,nwave_meas)
        CALL handle_nf90_error(status, "Could not inquire dimension 'nwave_meas'.")

        ! --- Get variables
        status = nf90_inq_varid(ncid, "wl_meas", varid_wl_meas)
        CALL handle_nf90_error(status, "Could not find variable 'wl_meas' in file.")
        status = nf90_get_var(ncid, varid_wl_meas, base_nn_data%wl_meas)
        CALL handle_nf90_error(status, "Could not read variable 'wl_meas' from file.")

        ! --- Close
        status = nf90_close(ncid)
        CALL handle_nf90_error(status, "Could not close netCDF file.")

        ! --- Store dimensions and other relevant properties
        base_nn_data%res_meas = &
                base_nn_data%wl_meas(2) - &
                base_nn_data%wl_meas(1)
        base_nn_data%n_wave_meas = SIZE(base_nn_data%wl_meas)

    END SUBROUTINE load_win_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_win_nn
    !> @brief Load neural network model for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] nn_path Path to directory containing NN related files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[out] nn_win Loaded neural network model for the specified window
    !------------------------------------------------------------------------------
    SUBROUTINE load_win_nn(nn_path, iwin, nn_win)
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
    !> @brief Links FTorch tensors to Fortran arrays in the base_nn_data_type
    !> @param[inout] base_nn_data The base_nn_data_type object to link tensors to arrays
    !------------------------------------------------------------------------------
    SUBROUTINE link_tensors_to_arrays(base_nn_data)
        TYPE(base_nn_data_type), INTENT(INOUT) :: base_nn_data

        CALL torch_tensor_from_array(base_nn_data%input_tensors(1), &
                                     base_nn_data%nn_inputs, &
                                     input_layout, torch_device)
        CALL torch_tensor_from_array(base_nn_data%output_tensors(1), &
                                     base_nn_data%spectrum, &
                                     output_layout, torch_device)

    END SUBROUTINE link_tensors_to_arrays


    !------------------------------------------------------------------------------
    ! SUBROUTINE handle_nf90_error
    !> @brief Handle netCDF error by printing message and stopping execution
    !> @param[in] status netCDF status code
    !> @param[in] msg Error message to display
    SUBROUTINE handle_nf90_error(status, msg)
        INTEGER, INTENT(IN) :: status
        CHARACTER(LEN=*), INTENT(IN) :: msg
        IF (status /= 0) THEN
            PRINT *, "NetCDF error: ", TRIM(nf90_strerror(status)), &
                     " Status code: ", status, &
                     " Message: ", TRIM(msg)
            STOP 1
        END IF
    END SUBROUTINE handle_nf90_error


    !------------------------------------------------------------------------------
    !  FUNCTION interp1d
    !> @brief Simple 1D linear interpolation function
    !> @param[in] x Known x values (monotonically increasing)
    !> @param[in] y Known y values
    !> @param[in] xq Query x value
    !> @param[out] yq Interpolated y value at xq
    !------------------------------------------------------------------------------
    FUNCTION interp1d(x, y, xq) RESULT(yq)
        DOUBLE PRECISION, INTENT(IN) :: x(:), y(:), xq
        DOUBLE PRECISION :: yq
        INTEGER :: j

        ! Find bracketing interval
        IF (xq <= x(1)) THEN
            yq = y(1)
        ELSE IF (xq >= x(size(x))) THEN
            yq = y(size(y))
        ELSE
            DO j = 1, size(x)-1
                IF (xq >= x(j) .AND. xq <= x(j+1)) THEN
                    yq = y(j) + (y(j+1)-y(j))*(xq-x(j))/(x(j+1)-x(j))
                    EXIT
                END IF
            END DO
        END IF
    END FUNCTION interp1d


END MODULE base_nn_forward_model_module
