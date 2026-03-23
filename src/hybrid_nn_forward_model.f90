!------------------------------------------------------------------------------
!  MODULE hybrid_nn_forward_model_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the subroutines of the S5 hybrid neural network 
!> forward model, including non-scattering calculations and ISRF convolution, for 
!> the NIR, SWIR1 and SWIR3 bands of the S5/UVNS spectrometer.
!------------------------------------------------------------------------------
MODULE hybrid_nn_forward_model_module
    USE ftorch
    USE netcdf
    USE hybrid_nn_data_type_module
    ! USE, INTRINSIC :: iso_fortran_env, only: wp => real32
    IMPLICIT NONE
    PRIVATE

    ! Methods implemented in this module
    PUBLIC :: forward_model_pixel_hybrid_nn
    PUBLIC :: forward_model_win_hybrid_nn
    PUBLIC :: forward_model_win_ns
    PUBLIC :: forward_model_win_ns_internal
    PUBLIC :: forward_model_win_nn_only
    PUBLIC :: forward_model_win_nn_forward_only
    PUBLIC :: forward_model_win_hybrid_nn_ns_only
    PUBLIC :: prepare_win_nn_inputs
    PUBLIC :: load_all_win_data
    PUBLIC :: load_hybrid_nn_win_data
    PUBLIC :: load_win_hybrid_nn
    PRIVATE :: link_tensors_to_arrays
    PRIVATE :: handle_nf90_error
    PRIVATE :: interp1d
    

    ! Predefined flag: Match wavelength channels directly or interpolate?
    ! Interpolation is slower but more flexible
    LOGICAL :: INTERPOLATE = .FALSE.

    ! Flag to indicate whether to use non-scattering input spectrum for NN 
    ! (True = hybrid NN, False = base NN)
    LOGICAL :: USE_NS_INPUT = .TRUE.

    ! Flag to indicate whether to use separate profile inputs for NN
    LOGICAL :: USE_PROFILE_INPUTS = .TRUE.

    ! Constants
    DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846d0

    ! Ftorch internal tensor layouts
    INTEGER, PARAMETER :: input_layout(1) = [1]
    INTEGER, PARAMETER :: proxy_layout(1) = [1]
    INTEGER, PARAMETER :: profile_layout(2) = [1, 2]
    INTEGER, PARAMETER :: output_layout(1) = [1]

    ! Ftorch internal tensor device
    INTEGER, PARAMETER :: torch_device = torch_kCPU

CONTAINS

    
    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_pixel_hybrid_nn
    !> @brief Forward model for one pixel (all bands) using the neural network with 
    !> non-scattering input
    !> @param[in] wl Wavelengths of all measurement channels of interest in nm (n_wave_total)
    !> @param[in] n_win Number of spectral windows
    !> @param[in/out] hybrid_nn_data_array Array of hybrid_nn_data_type for each spectral window (n_win)
    !> @param[out] radiances Output array of radiances at the specified wavelengths (n_wave_total)
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_pixel_hybrid_nn(wl, n_win, hybrid_nn_data_array, radiances)
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: wl
        INTEGER, INTENT(IN) :: n_win
        TYPE(hybrid_nn_data_type), DIMENSION(n_win), INTENT(INOUT) :: hybrid_nn_data_array
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
                n_wave_meas = hybrid_nn_data_array(iwin)%n_wave_meas

                IF (wl(iwave) >= hybrid_nn_data_array(iwin)%wl_meas(1) .AND. &
                    wl(iwave) <= hybrid_nn_data_array(iwin)%wl_meas(n_wave_meas)) THEN

                    ! This wavelength belongs to window iwin
                    found_wl = .TRUE.

                    ! If NN has not yet been run for this window, do so now
                    IF (.NOT. ran_nn_win(iwin)) THEN
                        CALL forward_model_win_hybrid_nn(hybrid_nn_data_array(iwin))
                        ran_nn_win(iwin) = .TRUE.
                    END IF

                    ! Now assign the corresponding output value to the spectrum
                    IF (INTERPOLATE) THEN
                        ! Interpolate to measurement wavelength grid
                        radiances(iwave) = interp1d( &
                                hybrid_nn_data_array(iwin)%wl_meas, &
                                hybrid_nn_data_array(iwin)%spectrum, &
                                wl(iwave))
                    ELSE
                        ! Directly match wavelength channels (faster, but requires exact match)
                        idx = INT((wl(iwave) - hybrid_nn_data_array(iwin)%wl_meas(1)) / &
                                               hybrid_nn_data_array(iwin)%res_meas) + 1
                        radiances(iwave) = hybrid_nn_data_array(iwin)%spectrum(idx)
                    END IF
                END IF
            END DO
            
            ! If wavelength was not found in any window, set to zero and show warning
            IF (.NOT. found_wl) THEN
                PRINT *, "Warning: wavelength ", wl(iwave), " outside NN forward model range."
                radiances(iwave) = 0.0d0
            END IF

        END DO

    END SUBROUTINE forward_model_pixel_hybrid_nn


    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_hybrid_nn
    !> @brief Forward model for one spectral window using the neural network with 
    !> non-scattering input
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding all necessary data

    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_hybrid_nn(hybrid_nn_data)
        ! Input & output parameters
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------

        IF(USE_NS_INPUT) THEN
            ! Run non-scattering forward model to get input spectrum for NN
            CALL forward_model_win_ns(hybrid_nn_data)
        END IF

        ! Apply NN correction to get final spectrum
        CALL forward_model_win_nn_only(hybrid_nn_data)

    END SUBROUTINE forward_model_win_hybrid_nn

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_nn_only
    !> @brief Run only NN-part of forward model for one spectral window
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding all necessary data
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_nn_only(hybrid_nn_data)
        ! Input & output parameters
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------

        ! Link Fortran arrays to FTorch tensors
        CALL link_tensors_to_arrays(hybrid_nn_data)

        ! Run NN for this window --> fills spectrum array in hybrid_nn_data
        CALL forward_model_win_nn_forward_only(hybrid_nn_data)
 
    END SUBROUTINE forward_model_win_nn_only

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_nn_forward_only
    !> @brief Run only NN forward pass for one spectral window, assuming that tensors are already linked
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding all necessary data
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_nn_forward_only(hybrid_nn_data)
        ! Input & output parameters
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------

        ! Run NN for this window --> fills spectrum array in hybrid_nn_data
        IF(USE_NS_INPUT) THEN
            IF(USE_PROFILE_INPUTS) THEN
                ! use non-scattering spectrum and profile inputs
                CALL torch_model_forward( &
                        hybrid_nn_data%nn_model, &
                        [hybrid_nn_data%input_tensors, &
                            hybrid_nn_data%proxy_tensors, &
                            hybrid_nn_data%profile_tensors], &
                        hybrid_nn_data%output_tensors)
            ELSE
                ! use non-scattering spectrum but no profile inputs
                CALL torch_model_forward( &
                        hybrid_nn_data%nn_model, &
                        [hybrid_nn_data%input_tensors, &
                            hybrid_nn_data%proxy_tensors], &
                        hybrid_nn_data%output_tensors)
            END IF
        ELSE
            IF(USE_PROFILE_INPUTS) THEN
                ! use profile inputs but no non-scattering spectrum
                CALL torch_model_forward( &
                        hybrid_nn_data%nn_model, &
                        [hybrid_nn_data%input_tensors, &
                            hybrid_nn_data%profile_tensors], &
                        hybrid_nn_data%output_tensors)
            ELSE
                ! use no non-scattering spectrum and no profile inputs (i.e. only pixel features)
                CALL torch_model_forward( &
                        hybrid_nn_data%nn_model, &
                        [hybrid_nn_data%input_tensors], &
                        hybrid_nn_data%output_tensors)
            END IF
        END IF
 
    END SUBROUTINE forward_model_win_nn_forward_only

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_hybrid_nn_ns_only
    !> @brief Forward model for one spectral window using the neural network code structure, 
    !> but only running the non-scattering forward model
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding all necessary data
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_hybrid_nn_ns_only(hybrid_nn_data)
        ! Input & output parameters
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------

        ! Run non-scattering forward model to get input spectrum for NN
        CALL forward_model_win_ns(hybrid_nn_data)

        ! Copy over radiances
        hybrid_nn_data%spectrum(:) = hybrid_nn_data%ns_spectrum(:)

    END SUBROUTINE forward_model_win_hybrid_nn_ns_only

    !------------------------------------------------------------------------------
    !  SUBROUTINE fwd_model_win_ns
    !> @brief Non-scattering forward model for a single spectral window
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object containing all necessary data 
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_ns(hybrid_nn_data)
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------

        ! Run non-scattering calulations for this window
        CALL forward_model_win_ns_internal( &
            hybrid_nn_data%n_wave_meas, &
            hybrid_nn_data%n_wave_fine, &
            hybrid_nn_data%n_ils_isrf, &
            hybrid_nn_data%cos_sza, &
            hybrid_nn_data%cos_vza, &
            hybrid_nn_data%wl_meas, &
            hybrid_nn_data%wl_fine, &
            hybrid_nn_data%res_fine, &
            hybrid_nn_data%xbdrf_wave_fine, &
            hybrid_nn_data%tau_abs_fine, &
            hybrid_nn_data%solar_irrad_fine, &
            hybrid_nn_data%dw_isrf, &
            hybrid_nn_data%resp_isrf, &
            hybrid_nn_data%ns_spectrum_fine, &
            hybrid_nn_data%ns_spectrum)

    END SUBROUTINE forward_model_win_ns

    !------------------------------------------------------------------------------
    !  SUBROUTINE fwd_model_win_ns_internal
    !> @brief Internal non-scattering calculations for a single spectral window
    !> @param[in] n_wave_meas Number of measurement wavelengths
    !> @param[in] n_wave_fine Number of fine wavelengths
    !> @param[in] n_isrf Number of ISRF points
    !> @param[in] cos_sza Cosine of solar zenith angle
    !> @param[in] cos_vza Cosine of viewing zenith angle
    !> @param[in] wl_meas Measurement wavelength grid (n_wave_meas)
    !> @param[in] wl_fine Fine wavelength grid (n_wave_fine)
    !> @param[in] res_fine Fine wavelength grid resolution (nm)
    !> @param[in] xbdrf_wave_fine Surface BRDF on fine grid (n_wave_fine)
    !> @param[in] tau_abs_fine Absorption optical depth on fine grid (n_wave_fine)
    !> @param[in] solar_irrad_fine Solar irradiance on fine grid (n_wave_fine)
    !> @param[in] dw_isrf ISRF delta wavelength grid (n_isrf, n_wave_meas)
    !> @param[in] resp_isrf ISRF response function (n_isrf, n_wave_meas)
    !> @param[out] ns_spectrum_fine Non-scattering spectrum on fine grid (n_wave_fine)
    !> @param[out] ns_spectrum Non-scattering spectrum on measurement grid (n_wave_meas)
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_ns_internal(n_wave_meas, n_wave_fine, n_isrf, &
            cos_sza, cos_vza, wl_meas, wl_fine, res_fine, xbdrf_wave_fine, &
            tau_abs_fine, solar_irrad_fine, dw_isrf, resp_isrf, &
            ns_spectrum_fine, ns_spectrum)
        INTEGER, INTENT(IN) :: n_wave_meas, n_wave_fine, n_isrf
        DOUBLE PRECISION, INTENT(IN) :: cos_sza, cos_vza
        DOUBLE PRECISION, DIMENSION(n_wave_meas), INTENT(IN) :: wl_meas
        DOUBLE PRECISION, DIMENSION(n_wave_fine), INTENT(IN) :: wl_fine
        DOUBLE PRECISION, INTENT(IN) :: res_fine
        DOUBLE PRECISION, DIMENSION(n_wave_fine), INTENT(IN) :: xbdrf_wave_fine
        DOUBLE PRECISION, DIMENSION(n_wave_fine), INTENT(IN) :: tau_abs_fine
        DOUBLE PRECISION, DIMENSION(n_wave_fine), INTENT(IN) :: solar_irrad_fine
        DOUBLE PRECISION, DIMENSION(n_isrf, n_wave_fine), INTENT(IN) :: dw_isrf
        DOUBLE PRECISION, DIMENSION(n_isrf, n_wave_fine), INTENT(IN) :: resp_isrf
        DOUBLE PRECISION, DIMENSION(n_wave_fine), INTENT(OUT) :: ns_spectrum_fine
        DOUBLE PRECISION, DIMENSION(n_wave_meas), INTENT(OUT) :: ns_spectrum
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwave, is, ie, nidx, i, j
        DOUBLE PRECISION :: dq
        DOUBLE PRECISION :: ws, we, dw
        DOUBLE PRECISION :: resp_linterp, total_resp
        DOUBLE PRECISION :: u
        DOUBLE PRECISION :: max_dw, min_dw
        !--------------------------------------------------------------------------

        u = 1.D0/DBLE(cos_sza) + 1.D0/DBLE(cos_vza)
        u = 1.D0/u

        ! calculate high-res radiance
        DO iwave = 1, n_wave_fine
            ns_spectrum_fine(iwave) = &
                xbdrf_wave_fine(iwave) * &
                DBLE(cos_sza)/PI * &
                EXP(-tau_abs_fine(iwave)/u)
        END DO
        ns_spectrum_fine = ns_spectrum_fine * solar_irrad_fine

        ! convolve with ISRF
        dw = res_fine
        DO iwave = 1, n_wave_meas

            ! get max and min delta wavelength for this channel
            max_dw = dw_isrf(n_isrf, iwave)
            min_dw = dw_isrf(1, iwave)

            ! determine wavelength range for convolution
            ws = wl_meas(iwave) + min_dw
            we = wl_meas(iwave) + max_dw

            ! determine indices for convolution (making sure they are within bounds)
            is = MAX(1, FLOOR(ABS(ws - wl_fine(1))/dw))
            ie = MIN(n_wave_fine, CEILING(ABS(we - wl_fine(1))/dw))
            nidx = ie - is + 1

            ! compute convolved radiance for this channel
            ns_spectrum(iwave) = 0.D0
            total_resp = 0.D0
            j = 1
            DO i = 1, nidx
                dq = wl_fine(is + i - 1) - wl_meas(iwave)

                ! Advance j until dq is bracketed
                DO WHILE (j < n_isrf)
                    IF (dq > dw_isrf(j+1, iwave)) THEN
                        j = j + 1
                    ELSE
                        EXIT
                    END IF
                END DO

                ! Linear interpolation
                IF (dq <= dw_isrf(1, iwave)) THEN
                    resp_linterp = resp_isrf(1, iwave)
                ELSE IF (dq >= dw_isrf(n_isrf, iwave)) THEN
                    resp_linterp = resp_isrf(n_isrf, iwave)
                ELSE
                    resp_linterp = resp_isrf(j, iwave) + &
                            (resp_isrf(j+1, iwave) - &
                            resp_isrf(j, iwave)) * &
                            (dq - dw_isrf(j, iwave)) / &
                            (dw_isrf(j+1, iwave) - &
                            dw_isrf(j, iwave))
                END IF

                total_resp = total_resp + resp_linterp
                ns_spectrum(iwave) = &
                        ns_spectrum(iwave) + &
                        ns_spectrum_fine(is + i - 1) * resp_linterp
            END DO

            ! normalize by total response
            ns_spectrum(iwave) = &
                    ns_spectrum(iwave) / total_resp

        END DO
 
    END SUBROUTINE forward_model_win_ns_internal

    !------------------------------------------------------------------------------
    ! SUBROUTINE prepare_win_nn_inputs
    !> @brief Prepare NN input tensor for a single spectral window
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[in] cos_sza Cosine of solar zenith angle
    !> @param[in] cos_vza Cosine of viewing zenith angle
    !> @param[in] rel_az Relative azimuth angle in radians
    !> @param[in] scat_angle Scattering angle in radians
    !> @param[in] p_surf Surface pressure in hPa
    !> @param[in] albedo Surface albedo
    !> @param[in] size_param Aerosol size parameter (exponent in powerlaw size distribution)
    !> @param[in] aod_550 Aerosol optical depth (AOD) at 550 nm
    !> @param[in] aer_height Aerosol layer height in m
    !> @param[in] h2o_col Water vapor column in molecules/m^2
    !> @param[in] co2_col CO2 column in molecules/m^2
    !> @param[in] ch4_col CH4 column in molecules/m^2
    !> @param[out] inputs Array of input parameters (n_nn_inputs)
    !------------------------------------------------------------------------------
    SUBROUTINE prepare_win_nn_inputs(iwin, cos_sza, cos_vza, rel_az, scat_angle, p_surf, albedo, &
                        size_param, aod_550,aer_height, h2o_col, co2_col, ch4_col, inputs)
        ! Input & output parameters
        INTEGER, INTENT(IN) :: iwin
        DOUBLE PRECISION, INTENT(IN) :: cos_sza
        DOUBLE PRECISION, INTENT(IN) :: cos_vza
        DOUBLE PRECISION, INTENT(IN) :: rel_az
        DOUBLE PRECISION, INTENT(IN) :: scat_angle
        DOUBLE PRECISION, INTENT(IN) :: p_surf
        DOUBLE PRECISION, INTENT(IN) :: albedo
        DOUBLE PRECISION, INTENT(IN) :: size_param
        DOUBLE PRECISION, INTENT(IN) :: aod_550
        DOUBLE PRECISION, INTENT(IN) :: aer_height
        DOUBLE PRECISION, INTENT(IN) :: h2o_col
        DOUBLE PRECISION, INTENT(IN) :: co2_col
        DOUBLE PRECISION, INTENT(IN) :: ch4_col
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: inputs
        !--------------------------------------------------------------------------
        ! Check input array size (different for each window)
        IF (iwin == 1) THEN
            ! assert that inputs has size 9 (0 trace gas inputs for NIR)
            IF(SIZE(inputs) /= 9) THEN
                ! stop and throw error
                CALL handle_hybrid_nn_error("inputs array must have size 9 for NIR window (iwin=1)")
            END IF
        ELSE IF (iwin == 2) THEN
            ! assert that inputs has size 12 (3 trace gas inputs for SWIR1)
            IF(SIZE(inputs) /= 12) THEN
                ! stop and throw error
                CALL handle_hybrid_nn_error("inputs array must have size 12 for SWIR1 window (iwin=2)")
            END IF
        ELSE IF (iwin == 3) THEN
            ! assert that inputs has size 11 (2 trace gas inputs for SWIR3)
            IF(SIZE(inputs) /= 11) THEN
                ! stop and throw error
                CALL handle_hybrid_nn_error("inputs array must have size 11 for SWIR3 window (iwin=3)")
            END IF
        END IF

        ! Fill input array
        inputs(1) = cos_sza
        inputs(2) = cos_vza
        inputs(3) = rel_az
        inputs(4) = scat_angle
        inputs(5) = p_surf
        inputs(6) = albedo
        inputs(7) = size_param
        inputs(8) = aod_550
        inputs(9) = aer_height
        IF (iwin == 2) THEN
            inputs(10) = h2o_col
            inputs(11) = co2_col
            inputs(12) = ch4_col
        ELSE IF (iwin == 3) THEN
            inputs(10) = h2o_col
            inputs(11) = ch4_col
        END IF

    END SUBROUTINE prepare_win_nn_inputs


    !------------------------------------------------------------------------------
    ! SUBROUTINE load_all_win_data
    !> @brief Load relevant data for all spectral windows (NIR, SWIR1, SWIR3)
    !> @param[in] win_data_path Path to directory containing window NetCDF files
    !> @param[in/out] hybrid_nn_data_array Array containing hybrid neural network data
    !------------------------------------------------------------------------------
    SUBROUTINE load_all_win_data(win_data_path, hybrid_nn_data_array)
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: win_data_path
        TYPE(hybrid_nn_data_type), DIMENSION(:), INTENT(INOUT) :: hybrid_nn_data_array
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwin
        INTEGER :: n_win
        !--------------------------------------------------------------------------

        ! Determine number of spectral windows
        n_win = SIZE(hybrid_nn_data_array)

        ! Load static data for each spectral window
        DO iwin = 1, n_win
            CALL load_hybrid_nn_win_data(win_data_path, iwin, hybrid_nn_data_array(iwin))
        END DO

    END SUBROUTINE load_all_win_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_hybrid_nn_win_data
    !> @brief Load static data for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] win_data_path Path to directory containing window NetCDF files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object containing all necessary data
    !------------------------------------------------------------------------------
    SUBROUTINE load_hybrid_nn_win_data(win_data_path, iwin, hybrid_nn_data)
        ! Input & output parameters
        CHARACTER(LEN=*), INTENT(IN) :: win_data_path
        INTEGER, INTENT(IN) :: iwin
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
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
            CALL handle_hybrid_nn_error("Invalid window index. Must be 1 (NIR), 2 (SWIR1), or 3 (SWIR3).")
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
        status = nf90_get_var(ncid, varid_wl_meas, hybrid_nn_data%wl_meas)
        CALL handle_nf90_error(status, "Could not read variable 'wl_meas' from file.")

        status = nf90_inq_varid(ncid, "wl_fine", varid_wl_fine)
        CALL handle_nf90_error(status, "Could not find variable 'wl_fine' in file.")
        status = nf90_get_var(ncid, varid_wl_fine, hybrid_nn_data%wl_fine)
        CALL handle_nf90_error(status, "Could not read variable 'wl_fine' from file.")

        status = nf90_inq_varid(ncid, "solar_irradiance", varid_solar_irrad)
        CALL handle_nf90_error(status, "Could not find variable 'solar_irradiance' in file.")
        status = nf90_get_var(ncid, varid_solar_irrad, hybrid_nn_data%solar_irrad_fine)
        CALL handle_nf90_error(status, "Could not read variable 'solar_irradiance' from file.")

        status = nf90_inq_varid(ncid, "dw_isrf", varid_dw)
        CALL handle_nf90_error(status, "Could not find variable 'dw_isrf' in file.")
        status = nf90_get_var(ncid, varid_dw, hybrid_nn_data%dw_isrf)
        CALL handle_nf90_error(status, "Could not read variable 'dw_isrf' from file.")

        status = nf90_inq_varid(ncid, "resp_isrf", varid_resp)
        CALL handle_nf90_error(status, "Could not find variable 'resp_isrf' in file.")
        status = nf90_get_var(ncid, varid_resp, hybrid_nn_data%resp_isrf)
        CALL handle_nf90_error(status, "Could not read variable 'resp_isrf' from file.")

        ! --- Close
        status = nf90_close(ncid)
        CALL handle_nf90_error(status, "Could not close netCDF file.")

        ! --- Store dimensions and other relevant properties
        hybrid_nn_data%res_meas = &
                hybrid_nn_data%wl_meas(2) - &
                hybrid_nn_data%wl_meas(1)
        hybrid_nn_data%res_fine = &
                hybrid_nn_data%wl_fine(2) - &
                hybrid_nn_data%wl_fine(1)
        hybrid_nn_data%n_wave_meas = SIZE(hybrid_nn_data%wl_meas)
        hybrid_nn_data%n_wave_fine = SIZE(hybrid_nn_data%wl_fine)
        hybrid_nn_data%n_ils_isrf = SIZE(hybrid_nn_data%dw_isrf, 1)

    END SUBROUTINE load_hybrid_nn_win_data

    !------------------------------------------------------------------------------
    ! SUBROUTINE load_win_hybrid_nn
    !> @brief Load neural network model for one spectral window (NIR, SWIR1, or SWIR3)
    !> @param[in] nn_path Path to directory containing NN related files
    !> @param[in] iwin Spectral window index (1 = NIR, 2 = SWIR1, 3 = SWIR3)
    !> @param[out] nn_win Loaded neural network model for the specified window
    !------------------------------------------------------------------------------
    SUBROUTINE load_win_hybrid_nn(nn_path, iwin, nn_win)
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
            CALL handle_hybrid_nn_error("Invalid window index. Must be 1 (NIR), 2 (SWIR1), or 3 (SWIR3).")
        END IF

        ! Load neural network model for the specified spectral window
        CALL torch_model_load(nn_win, TRIM(nn_path) // TRIM(win_str) // '.pt', torch_kCPU)

    END SUBROUTINE load_win_hybrid_nn

    !------------------------------------------------------------------------------
    !> @brief Links FTorch tensors to Fortran arrays in the hybrid_nn_data_type
    !> @param[inout] hybrid_nn_data The hybrid_nn_data_type object to link tensors to arrays
    !------------------------------------------------------------------------------
    SUBROUTINE link_tensors_to_arrays(hybrid_nn_data)
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data

        CALL torch_tensor_from_array(hybrid_nn_data%input_tensors(1), &
                                     hybrid_nn_data%nn_inputs, &
                                     input_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%proxy_tensors(1), &
                                     hybrid_nn_data%ns_spectrum, &
                                     proxy_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%profile_tensors(1), &
                                     hybrid_nn_data%nn_profile_inputs, &
                                     profile_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%output_tensors(1), &
                                     hybrid_nn_data%spectrum, &
                                     output_layout, torch_device)

    END SUBROUTINE link_tensors_to_arrays


    !------------------------------------------------------------------------------
    ! SUBROUTINE handle_hybrid_nn_error
    !> @brief Handle hybrid NN error by printing message and stopping execution
    !> @param[in] msg Error message to display
    !------------------------------------------------------------------------------
    SUBROUTINE handle_hybrid_nn_error(msg)
        CHARACTER(LEN=*), INTENT(IN) :: msg
        PRINT *, "Error: ", TRIM(msg)
        STOP 1
    END SUBROUTINE handle_hybrid_nn_error
    
    !------------------------------------------------------------------------------
    ! SUBROUTINE handle_nf90_error
    !> @brief Handle netCDF error by printing message and stopping execution
    !> @param[in] status netCDF status code
    !> @param[in] msg Error message to display
    !------------------------------------------------------------------------------
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


END MODULE hybrid_nn_forward_model_module
