!------------------------------------------------------------------------------
!  MODULE s5nn_core
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the core implementation of the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE s5nn_core
    USE mod_s5nn_derived_type, ONLY: N_WIN, MIN_IDX_WAVE_FINE, MAX_IDX_WAVE_FINE
    IMPLICIT NONE

    ! Constants
    DOUBLE PRECISION, PARAMETER :: PI = 3.14159265358979323846d0

    ! Ftorch internal tensor layouts
    INTEGER, PARAMETER :: input_layout(1) = [1]
    INTEGER, PARAMETER :: proxy_layout(1) = [1]
    INTEGER, PARAMETER :: output_layout(1) = [1]

    ! Ftorch internal tensor device
    INTEGER, PARAMETER :: torch_device = torch_kCPU

    ! Public subroutines implemented in this module
    PUBLIC :: forward_model_win_nn_hybrid
    PUBLIC :: forward_model_win_nn_base
    PUBLIC :: forward_model_win_ns

    ! Everything else in this module is private
    PRIVATE

    !------------------------------------------------------------------------------
    CONTAINS


    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_nn_hybrid
    !> @brief Forward model for one spectral window using the neural network with 
    !> non-scattering input
    !> @param[in/out] win_data_i The s5nn_win_data object for the spectral window
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_nn_hybrid(win_data_i)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        ! Input & output parameters
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data_i
        !--------------------------------------------------------------------------

        ! Run non-scattering forward model to get input spectrum for NN
        CALL forward_model_win_ns(win_data_i)

        ! Link Fortran arrays to FTorch tensors
        CALL link_tensors_to_arrays(win_data_i)

        ! Run NN for this window --> fills spectrum array in win_data_i
        CALL torch_model_forward( &
                win_data_i%nn_model, &
                [win_data_i%input_tensors, win_data_i%proxy_tensors], &
                win_data_i%output_tensors)

    END SUBROUTINE forward_model_win_nn_hybrid

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_win_nn_base
    !> @brief Forward model for one spectral window using the base neural network
    !> @param[in/out] win_data_i The s5nn_win_data object for the spectral window
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_nn_base(win_data_i)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        ! Input & output parameters
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data_i
        !--------------------------------------------------------------------------

        ! Link Fortran arrays to FTorch tensors
        CALL link_tensors_to_arrays(win_data_i)

        ! Run NN for this window --> fills spectrum array in win_data_i
        CALL torch_model_forward( &
                win_data_i%nn_model, &
                [win_data_i%input_tensors], &
                win_data_i%output_tensors)

    END SUBROUTINE forward_model_win_nn_base

    !------------------------------------------------------------------------------
    !  SUBROUTINE fwd_model_win_ns
    !> @brief Non-scattering forward model for a single spectral window
    !> @param[in/out] win_data_i The s5nn_win_data object for the spectral window
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_win_ns(win_data_i)
        IMPLICIT NONE
        !--------------------------------------------------------------------------
        ! Input & output parameters
        TYPE(s5nn_win_data), INTENT(INOUT) :: win_data_i
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: n_wave_meas, n_wave_fine, n_isrf
        INTEGER :: iwave, is, ie, nidx, i, j
        DOUBLE PRECISION :: dq
        DOUBLE PRECISION :: ws, we, dw
        DOUBLE PRECISION :: resp_linterp, total_resp
        DOUBLE PRECISION :: u
        DOUBLE PRECISION :: max_dw, min_dw
        !--------------------------------------------------------------------------

        n_wave_meas = win_data_i%n_wave_meas
        n_wave_fine = win_data_i%n_wave_fine
        n_isrf = win_data_i%n_ils_isrf

        u = 1.D0/DBLE(win_data_i%cos_sza) + 1.D0/DBLE(win_data_i%cos_vza)
        u = 1.D0/u

        ! calculate high-res radiance
        DO iwave = 1, n_wave_fine
            win_data_i%ns_spectrum_fine(iwave) = &
                win_data_i%xbdrf_wave_fine(iwave) * &
                DBLE(win_data_i%cos_sza)/PI * &
                EXP(-win_data_i%tau_abs_fine(iwave)/u)
        END DO
        win_data_i%ns_spectrum_fine = &
                win_data_i%ns_spectrum_fine * &
                win_data_i%solar_irrad_fine

        ! convolve with ISRF
        dw = win_data_i%res_fine
        DO iwave = 1, n_wave_meas

            ! get max and min delta wavelength for this channel
            max_dw = win_data_i%dw_isrf(n_isrf, iwave)
            min_dw = win_data_i%dw_isrf(1, iwave)

            ! determine wavelength range for convolution
            ws = win_data_i%wl_meas(iwave) + min_dw
            we = win_data_i%wl_meas(iwave) + max_dw

            ! determine indices for convolution (making sure they are within bounds)
            is = MAX(1, FLOOR(ABS(ws - win_data_i%wl_fine(1))/dw))
            ie = MIN(n_wave_fine, CEILING(ABS(we - win_data_i%wl_fine(1))/dw))
            nidx = ie - is + 1

            ! compute convolved radiance for this channel
            win_data_i%ns_spectrum(iwave) = 0.D0
            total_resp = 0.D0
            j = 1
            DO i = 1, nidx
                dq = win_data_i%wl_fine(is + i - 1) - win_data_i%wl_meas(iwave)

                ! Advance j until dq is bracketed
                DO WHILE (j < n_isrf)
                    IF (dq > win_data_i%dw_isrf(j+1, iwave)) THEN
                        j = j + 1
                    ELSE
                        EXIT
                    END IF
                END DO

                ! Linear interpolation
                IF (dq <= win_data_i%dw_isrf(1, iwave)) THEN
                    resp_linterp = win_data_i%resp_isrf(1, iwave)
                ELSE IF (dq >= win_data_i%dw_isrf(n_isrf, iwave)) THEN
                    resp_linterp = win_data_i%resp_isrf(n_isrf, iwave)
                ELSE
                    resp_linterp = win_data_i%resp_isrf(j, iwave) + &
                            (win_data_i%resp_isrf(j+1, iwave) - &
                            win_data_i%resp_isrf(j, iwave)) * &
                            (dq - win_data_i%dw_isrf(j, iwave)) / &
                            (win_data_i%dw_isrf(j+1, iwave) - &
                            win_data_i%dw_isrf(j, iwave))
                END IF

                total_resp = total_resp + resp_linterp
                win_data_i%ns_spectrum(iwave) = &
                        win_data_i%ns_spectrum(iwave) + &
                        win_data_i%ns_spectrum_fine(is + i - 1) * resp_linterp
            END DO

            ! normalize by total response
            win_data_i%ns_spectrum(iwave) = &
                    win_data_i%ns_spectrum(iwave) / total_resp

        END DO
 
    END SUBROUTINE forward_model_win_ns

    !------------------------------------------------------------------------------
    !> @brief Links FTorch tensors to Fortran arrays in the s5nn_win_data_type
    !> @param[inout] win_data_i The s5nn_win_data object for the spectral window
    !------------------------------------------------------------------------------
    SUBROUTINE link_tensors_to_arrays(win_data_i)
        TYPE(s5nn_win_data_type), INTENT(INOUT) :: win_data_i

        CALL torch_tensor_from_array(win_data_i%input_tensors(1), &
                                     win_data_i%nn_inputs, &
                                     input_layout, torch_device)
        CALL torch_tensor_from_array(win_data_i%proxy_tensors(1), &
                                     win_data_i%ns_spectrum, &
                                     proxy_layout, torch_device)
        CALL torch_tensor_from_array(win_data_i%output_tensors(1), &
                                     win_data_i%spectrum, &
                                     output_layout, torch_device)

    END SUBROUTINE link_tensors_to_arrays

END MODULE s5nn_core