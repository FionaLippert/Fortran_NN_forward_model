!------------------------------------------------------------------------------
!  MODULE bridge_grasp_s5nn
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the bridge between GRASP and the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE bridge_grasp_s5nn
    USE mod_s5nn_derived_type, ONLY: N_WIN, MIN_IDX_WAVE_FINE, MAX_IDX_WAVE_FINE, & 
                                     forward_model_characteristics_s5nn
    IMPLICIT NONE


    ! Public subroutines implemented in this module
    PUBLIC :: update_nn_data_from_forw_s5nn
    PUBLIC :: update_forw_s5nn_from_nn_data

    ! Everything else in this module is private
    PRIVATE

    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Updates internal nn_data based on fw_s5nn characteristics
    !> @param[inout] fw_s5nn The forward_model_characteristics_s5nn data structure
    !------------------------------------------------------------------------------
    SUBROUTINE update_nn_data_from_forw_s5nn(fw_s5nn)
        IMPLICIT NONE
        USE mod_alloc_s5nn, ONLY: nn_data
        !------------------------------------------------------------------------------
        TYPE(forward_model_characteristics_s5nn), INTENT(INOUT) :: fw_s5nn
        !------------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwin, imin, imax
        !------------------------------------------------------------------------------

        ! Update data for each spectral window
        DO iwin = 1, N_WIN
            ! Determine fine wavelength index range for this window
            imin = MIN_IDX_WAVE_FINE(iwin)
            imax = MAX_IDX_WAVE_FINE(iwin)

            ! Update NN input parameters for this window
            CALL prepare_win_nn_inputs( &
                    iwin, &
                    fw_s5nn%cos_sza, 
                    fw_s5nn%cos_vza, 
                    fw_s5nn%rel_az, 
                    fw_s5nn%scat_angle, &
                    fw_s5nn%avg_albedo(iwin), &
                    fw_s5nn%angstrom_exponent, &
                    fw_s5nn%aod_550, &
                    fw_s5nn%aer_height, &
                    fw_s5nn%h2o_column, &
                    fw_s5nn%co2_column, &
                    fw_s5nn%ch4_column, &
                    nn_data%win_data(iwin)%nn_inputs)

            ! Update data needed for non-scattering calculations
            nn_data%cos_sza = fw_s5nn%cos_sza
            nn_data%cos_vza = fw_s5nn%cos_vza
            nn_data%xbdrf_wave_fine(:) = fw_s5nn%xbdrf_wave_fine(imin:imax)
            nn_data%tau_abs_fine(:) = fw_s5nn%tau_abs_fine(imin:imax)

        END DO

    END SUBROUTINE update_nn_data_from_forw_s5nn

    !------------------------------------------------------------------------------
    !> @brief Updates fw_s5nn characteristics based on internal nn_data
    !> @param[inout] fw_s5nn The forward_model_characteristics_s5nn data structure
    !------------------------------------------------------------------------------
    SUBROUTINE update_forw_s5nn_from_nn_data(fw_s5nn)
        IMPLICIT NONE
        USE mod_alloc_s5nn, ONLY: nn_data
        !-------------------------------------------------------------------------------
        TYPE(forward_model_characteristics_s5nn), INTENT(INOUT) :: fw_s5nn
        !-------------------------------------------------------------------------------

        ! For each spectral window, write radiances from nn_data to fw_s5nn
        DO iwin = 1, N_WIN
            ! Determine fine wavelength index range for this window
            imin = MIN_IDX_WAVE_FINE(iwin)
            imax = MAX_IDX_WAVE_FINE(iwin)

            ! Update radiances for this window
            fw_s5nn%radiances(imin:imax) = nn_data%win_data(iwin)%spectrum(:)
        END DO

    END SUBROUTINE update_forw_s5nn_from_nn_data

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



END MODULE bridge_grasp_s5nn