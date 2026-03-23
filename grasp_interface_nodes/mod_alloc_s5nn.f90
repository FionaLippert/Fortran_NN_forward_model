!------------------------------------------------------------------------------
!  MODULE mod_alloc_s5nn
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains allocation and deallocation subroutines for the 
!> S5 neural network forward model.
!------------------------------------------------------------------------------
MODULE mod_alloc_s5nn
    USE mod_s5nn_derived_type, ONLY: N_WIN, N_IN, N_WAVE_MEAS, N_WAVE_FINE, N_ILS_ISRF, &
                                     s5nn_data
    ! TODO: where is retr_input_settings defined?
    IMPLICIT NONE

    !--- Public subroutines implemented in this module
    PUBLIC :: allocate_forward_model_s5nn
    PUBLIC :: deallocate_forward_model_s5nn

    !--- Public variables defined in this module
    PUBLIC :: nn_data

    ! Everything else in this module is private
    PRIVATE

    !--- General data structure used for internal NN forward model computations
    TYPE(s5nn_data) :: nn_data

CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Allocates memory for the S5 neural network forward model
    !> @param[inout] RIN The GRASP input configuration data structure
    !------------------------------------------------------------------------------
    SUBROUTINE allocate_forward_model_s5nn(RIN)  
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        ! Inputs & outputs
        TYPE(retr_input_settings), INTENT(INOUT) :: RIN
        !------------------------------------------------------------------------------
        ! Local variables
        INTEGER :: iwin
        !------------------------------------------------------------------------------

        ! TODO: Set up N_WIN etc. based on RIN ?

        ! Allocate data for all spectral windows
        DO iwin = 1, N_WIN
            CALL allocate_s5nn_win_data(nn_data%win_data(iwin), &
                    N_IN(iwin), N_WAVE_MEAS(iwin), N_WAVE_FINE(iwin), N_ILS_ISRF(iwin))
        END DO

    END SUBROUTINE allocate_forward_model_s5nn

    !------------------------------------------------------------------------------
    !> @brief Deallocates memory for the S5 neural network forward model
    !> @param[inout] RIN The GRASP input configuration data structure
    !------------------------------------------------------------------------------
    SUBROUTINE deallocate_forward_model_s5nn(RIN)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        ! Inputs & outputs
        TYPE(retr_input_settings), INTENT(INOUT) :: RIN
        !------------------------------------------------------------------------------

        ! Deallocate data for all spectral windows
        DO iwin = 1, N_WIN
            CALL deallocate_s5nn_win_data(nn_data%win_data(iwin))
        END DO

    END SUBROUTINE deallocate_forward_model_s5nn


    !------------------------------------------------------------------------------
    !> @brief Allocates memory for one spectral window
    !> @param[inout] win_data The s5nn_win_data object to allocate memory for
    !> @param[in] n_inputs Number of NN input features
    !> @param[in] n_wave_meas Number of measurement wavelengths
    !> @param[in] n_wave_fine Number of fine wavelengths
    !> @param[in] n_ils_isrf Number of ISRF points
    !------------------------------------------------------------------------------
    SUBROUTINE allocate_s5nn_win_data(win_data, n_inputs, n_wave_meas, n_wave_fine, n_ils_isrf)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data
        INTEGER, INTENT(IN) :: n_inputs, n_wave_meas, n_wave_fine, n_ils_isrf
        !------------------------------------------------------------------------------

        ALLOCATE(win_data%nn_inputs(n_inputs))
        ALLOCATE(win_data%ns_spectrum(n_wave_meas))
        ALLOCATE(win_data%spectrum(n_wave_meas))
        ALLOCATE(win_data%wl_meas(n_wave_meas))
        ALLOCATE(win_data%wl_fine(n_wave_fine))
        ALLOCATE(win_data%resp_isrf(n_ils_isrf, n_wave_meas))
        ALLOCATE(win_data%dw_isrf(n_ils_isrf, n_wave_meas))
        ALLOCATE(win_data%solar_irrad_fine(n_wave_fine))
        ALLOCATE(win_data%xbdrf_wave_fine(n_wave_fine))
        ALLOCATE(win_data%tau_abs_fine(n_wave_fine))
        ALLOCATE(win_data%ns_spectrum_fine(n_wave_fine))

    END SUBROUTINE allocate_s5nn_win_data

    !------------------------------------------------------------------------------
    !> @brief Deallocates memory for one spectral window
    !> @param[inout] win_data The s5nn_win_data object to deallocate memory for
    !------------------------------------------------------------------------------
    SUBROUTINE deallocate_s5nn_win_data(win_data)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data
        !------------------------------------------------------------------------------

        IF (ALLOCATED(win_data%nn_inputs)) DEALLOCATE(win_data%nn_inputs)
        IF (ALLOCATED(win_data%ns_spectrum)) DEALLOCATE(win_data%ns_spectrum)
        IF (ALLOCATED(win_data%spectrum)) DEALLOCATE(win_data%spectrum)
        IF (ALLOCATED(win_data%wl_meas)) DEALLOCATE(win_data%wl_meas)
        IF (ALLOCATED(win_data%wl_fine)) DEALLOCATE(win_data%wl_fine)
        IF (ALLOCATED(win_data%resp_isrf)) DEALLOCATE(win_data%resp_isrf)
        IF (ALLOCATED(win_data%dw_isrf)) DEALLOCATE(win_data%dw_isrf)
        IF (ALLOCATED(win_data%solar_irrad_fine)) DEALLOCATE(win_data%solar_irrad_fine)
        IF (ALLOCATED(win_data%xbdrf_wave_fine)) DEALLOCATE(win_data%xbdrf_wave_fine)
        IF (ALLOCATED(win_data%tau_abs_fine)) DEALLOCATE(win_data%tau_abs_fine)
        IF (ALLOCATED(win_data%ns_spectrum_fine)) DEALLOCATE(win_data%ns_spectrum_fine)

        ! delete FTorch tensors and model
        CALL torch_delete(win_data%input_tensors(1))
        CALL torch_delete(win_data%proxy_tensors(1))
        CALL torch_delete(win_data%output_tensors(1))
        CALL torch_delete(win_data%nn_model)

    END SUBROUTINE deallocate_s5nn_win_data
