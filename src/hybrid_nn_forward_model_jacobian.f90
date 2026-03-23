!------------------------------------------------------------------------------
!  MODULE hybrid_nn_forward_model_jacobian_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the subroutines for computing Jacobians of the 
!> S5 hybrid neural network forward model, using finite differences for the NN
!> part and analytical derivatives for the non-scattering part.
!------------------------------------------------------------------------------
MODULE hybrid_nn_forward_model_jacobian_module
    USE ftorch
    USE hybrid_nn_data_type_module, ONLY: hybrid_nn_data_type
    USE hybrid_nn_forward_model_module, ONLY: forward_model_win_nn_only, &
                                              forward_model_win_nn_forward_only, &
                                              forward_model_win_ns, &
                                              forward_model_win_ns_internal, &
                                              forward_model_win_hybrid_nn, &
                                              USE_NS_INPUT, USE_PROFILE_INPUTS, PI, &
                                              input_layout, proxy_layout, &
                                              profile_layout, output_layout, &
                                              torch_device
    IMPLICIT NONE
    PRIVATE

    ! Methods implemented in this module
    PUBLIC :: forward_model_kernel_win_hybrid_nn
    PUBLIC :: forward_model_kernel_win_implicit_hybrid_nn
    PUBLIC :: forward_model_kernel_win_hybrid_nn_ns_only
    PUBLIC :: apply_perturbation
    PUBLIC :: finite_difference_win_hybrid_nn
    PUBLIC :: forward_model_kernel_win_ns
    PRIVATE :: link_tensors_to_perturbed_arrays
    PRIVATE :: compute_perturbed_aod_550

CONTAINS


    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_kernel_win_hybrid_nn
    !> @brief NN forward model and derivatives for one spectral window. Derivatives are obtained by 
    !> combining analytical non-scattering derivatives (if applicable) with finite difference derivatives of the NN part.
    !> @param[in] delta_values Array of perturbations for finite difference derivatives (n_nn_inputs)
    !> @param[in] cabs_mol_fine Fine grid of aerosol absorption coefficients (n_wave_fine)
    !> @param[in] dvmol_prior Prior distribution of vertical profiles (n_wave_fine)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding data for the spectral window
    !------------------------------------------------------------------------------

    ! Note: Assuming profile scaling, it is dI/dc_mol = dI/dv_mol * v_mol_ref (so scaled by total reference column)

    SUBROUTINE forward_model_kernel_win_hybrid_nn( &
                        delta_values, &
                        cabs_mol_fine, &
                        dvmol_prior, &
                        hybrid_nn_data, &
                        )
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: delta_values
        DOUBLE PRECISION, DIMENSION(:, :, :), INTENT(IN) :: cabs_mol_fine
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(IN) :: dvmol_prior
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: n_mol
        INTEGER :: imol
        !---------------------------------------------------------------------------
 
        n_mol = SIZE(dvmol_prior, 2)  ! number of gases

        IF(USE_NS_INPUT) THEN
            ! If using non-scattering input to the NN, we need to run the non-scattering 
            ! forward model first to get the input spectrum and derivatives
            CALL forward_model_kernel_win_ns( &
                                cabs_mol_fine, &
                                dvmol_prior, &
                                hybrid_nn_data)
            ! For all derivatives, keep non-scattering spectrum fixed and use analytical
            ! derivatives computed above to combine with NN derivatives if needed
            hybrid_nn_data%ns_spectrum_perturbed(:) = hybrid_nn_data%ns_spectrum(:)
        END IF

        ! Get baseline forward model output for unperturbed state
        CALL forward_model_win_nn_only(hybrid_nn_data)

        ! Now, compute numerical derivatives for each state vector parameter
        ! Start with avg albedo
        CALL apply_perturbation(hybrid_nn_data, delta_values(6), 6)
        CALL finite_difference_win_hybrid_nn(delta_values(6), &
                hybrid_nn_data, &
                hybrid_nn_data%der_albedo)

        IF(USE_NS_INPUT) THEN
            ! Combine the NN derivative with the non-scattering derivative through the product rule
            ! where f_nn is NN-based correction factor, s.t. I_nn = I_ns * f_nn
            ! dI_nn/dx = f_nn * dI_ns/dx + I_ns * df_nn/dx
            ! Since the FTorch NN outputs the combined radiance I_nn = I_ns * f_nn,
            ! we need to omit the multiplication with I_ns in the NN derivative part
            hybrid_nn_data%der_albedo = (hybrid_nn_data%spectrum / hybrid_nn_data%ns_spectrum) &
                                            * hybrid_nn_data%ns_der_albedo + hybrid_nn_data%der_albedo
        END IF

        ! Then aerosol parameters
        ! (aerosol optical depth, layer height, size parameter)
        ! Note that the aerosol derivatives don't need to be adjusted if using non-scattering inputs, 
        ! because the finite difference approximation already includes the multiplication
        ! with the non-scattering spectrum I_ns
        CALL apply_perturbation(hybrid_nn_data, delta_values(8), 8)
        CALL finite_difference_win_hybrid_nn(delta_values(8), &
                hybrid_nn_data, &
                hybrid_nn_data%der_aod)
        CALL apply_perturbation(hybrid_nn_data, delta_values(9), 9)
        CALL finite_difference_win_hybrid_nn(delta_values(9), &
                hybrid_nn_data, &
                hybrid_nn_data%der_aer_height)

        ! Now, aerosol size paramter
        CALL apply_perturbation(hybrid_nn_data, delta_values(7), 7)
        ! Note that the AOD input is dependent on the size parameter, 
        ! so we need to apply the perturbation to the size parameter input 
        ! and then recompute the AOD perturbation to ensure that the AOD input 
        ! so that it is consistent with the perturbed size parameter
        CALL compute_perturbed_aod_550( &
            hybrid_nn_data, &
            hybrid_nn_data%nn_inputs_perturbed(7), & ! perturbed size parameter input
            hybrid_nn_data%nn_inputs_perturbed(8), & ! resulting perturbed AOD input
        )
        ! now, compute finite difference derivative for size parameter input
        CALL finite_difference_win_hybrid_nn(delta_values(7), &
                hybrid_nn_data, &
                hybrid_nn_data%der_size_param)

        ! Then gas columns (loop over each gas)
        DO imol = 1, n_mol
            CALL apply_perturbation(hybrid_nn_data, delta_values(9 + imol), 9 + imol)
            CALL finite_difference_win_hybrid_nn(delta_values(9 + imol), &
                    hybrid_nn_data, &
                    hybrid_nn_data%der_c_mol(:, imol))

            ! scale by reference column to get dI/dc_mol
            hybrid_nn_data%der_c_mol(:, imol) = hybrid_nn_data%der_c_mol(:, imol) * SUM(dvmol_prior(:, imol))

            IF(USE_NS_INPUT) THEN
                ! Combine with non-scattering derivative through product rule
                ! dI/dx = I_nn * dI_ns/dx + I_ns * dI_nn/dx
                ! Same as for albedo, we need to omit the multiplication with I_ns in the NN derivative part
                hybrid_nn_data%der_c_mol(:, imol) = (hybrid_nn_data%spectrum / hybrid_nn_data%ns_spectrum) &
                                        * hybrid_nn_data%ns_der_c_mol(:, imol) + hybrid_nn_data%der_c_mol(:, imol)
            END IF
        END DO

    END SUBROUTINE forward_model_kernel_win_hybrid_nn

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_kernel_win_implicit_hybrid_nn
    !> @brief NN forward model and derivatives for one spectral window. Effects of 
    !> non-scattering inputs on forward model derivatives are computed implicitly
    !> through finite difference perturbations, rather than using the analytical derivatives.
    !> @param[in] delta_values Array of perturbations for finite difference derivatives (n_nn_inputs)
    !> @param[in] cabs_mol_fine Fine grid of molecular absorption coefficients (n_wave_fine)
    !> @param[in] dvmol_prior Prior distribution of vertical profiles (n_wave_fine)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding data for the spectral window
    !------------------------------------------------------------------------------

    ! Note: Assuming profile scaling, it is dI/dc_mol = dI/dv_mol * v_mol_ref (so scaled by total reference column)

    SUBROUTINE forward_model_kernel_win_implicit_hybrid_nn( &
                        delta_values, &
                        cabs_mol_fine, &
                        dvmol_prior, &
                        hybrid_nn_data, &
                        )
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: delta_values
        DOUBLE PRECISION, DIMENSION(:, :, :), INTENT(IN) :: cabs_mol_fine
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(IN) :: dvmol_prior
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: n_mol
        INTEGER :: imol
        DOUBLE PRECISION :: vmol_prior, c_mol, c_mol_perturbed
        !---------------------------------------------------------------------------
 
        n_mol = SIZE(dvmol_prior, 2)  ! number of gases

        IF(USE_NS_INPUT) THEN
            ! If using non-scattering input to the NN, we need to run the non-scattering 
            ! forward model first to get the input spectrum (no derivatives needed here)
            CALL forward_model_win_ns(hybrid_nn_data)
        END IF

        ! Get baseline forward model output for unperturbed state
        CALL forward_model_win_nn_only(hybrid_nn_data)

        ! Now, compute numerical derivatives for each state vector parameter
        ! Start with aerosol parameters
        ! (aerosol optical depth, layer height, size parameter)
        ! Note that the non-scattering spectrum does not depend on these parameters,
        ! so we do not need to recompute it for the perturbed values
        hybrid_nn_data%ns_spectrum_perturbed(:) = &
                hybrid_nn_data%ns_spectrum(:)
        CALL apply_perturbation(hybrid_nn_data, delta_values(8), 8)
        CALL finite_difference_win_hybrid_nn(delta_values(8), &
                hybrid_nn_data, &
                hybrid_nn_data%der_aod)
        CALL apply_perturbation(hybrid_nn_data, delta_values(9), 9)
        CALL finite_difference_win_hybrid_nn(delta_values(9), &
                hybrid_nn_data, &
                hybrid_nn_data%der_aer_height)

        ! For aerosol size paramter, combine direct derivatives with derivatives through AOD
        CALL apply_perturbation(hybrid_nn_data, delta_values(7), 7)
        CALL compute_perturbed_aod_550( &
            hybrid_nn_data, &
            hybrid_nn_data%nn_inputs_perturbed(7), & ! perturbed size parameter input
            hybrid_nn_data%nn_inputs_perturbed(8), & ! resulting perturbed AOD input
        )
        ! now, compute finite difference derivative for size parameter input
        CALL finite_difference_win_hybrid_nn(delta_values(7), &
                hybrid_nn_data, &
                hybrid_nn_data%der_size_param)

        ! Now, albedo (assuming constant albedo in the window)
        IF(USE_NS_INPUT) THEN
            ! Get perturbed non-scattering spectrum by scaling with relative perturbed albedo
            hybrid_nn_data%ns_spectrum_perturbed(:) = hybrid_nn_data%ns_spectrum(:) * &
                    (1.0D0 + delta_values(6) / hybrid_nn_data%nn_inputs(6))
        END IF
        ! Compute numerical derivatives for albedo based on perturbed 
        ! non-scattering spectrum (if applicable) and perturbed albedo
        CALL apply_perturbation(hybrid_nn_data, delta_values(6), 6)
        CALL finite_difference_win_hybrid_nn(delta_values(6), &
                hybrid_nn_data, &
                hybrid_nn_data%der_albedo)

        ! Then gas columns (loop over each gas)
        DO imol = 1, n_mol
            IF(USE_NS_INPUT) THEN
                ! Compute perturbed molecular absorption optical depth 
                ! for this gas based on perturbed column
                vmol_prior = SUM(dvmol_prior(:, imol))
                c_mol = hybrid_nn_data%nn_inputs(9 + imol) / vmol_prior
                c_mol_perturbed = (hybrid_nn_data%nn_inputs(9 + imol) + &
                                    delta_values(9 + imol)) / vmol_prior
                hybrid_nn_data%tau_abs_fine_perturbed(:) = &
                    hybrid_nn_data%tau_abs_fine(:)

                DO iwave = 1, hybrid_nn_data%n_wave_fine
                    hybrid_nn_data%tau_abs_fine_perturbed(iwave) = &
                        hybrid_nn_data%tau_abs_fine(iwave) + &
                        (c_mol_perturbed - c_mol) * &
                        SUM(cabs_mol_fine(iwave, :, imol) * dvmol_prior(:, imol))
                END DO

                ! Then, compute corresponding perturbed non-scattering spectrum
                ! Run non-scattering calculations with perturbed gas input
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
                    hybrid_nn_data%tau_abs_fine_perturbed, &
                    hybrid_nn_data%solar_irrad_fine, &
                    hybrid_nn_data%dw_isrf, &
                    hybrid_nn_data%resp_isrf, &
                    hybrid_nn_data%ns_spectrum_fine, &
                    hybrid_nn_data%ns_spectrum_perturbed)
            END IF

            ! Now, apply pertubation to NN inputs and compute finite difference derivative
            CALL apply_perturbation(hybrid_nn_data, delta_values(9 + imol), 9 + imol)
            CALL finite_difference_win_hybrid_nn(delta_values(9 + imol), &
                    hybrid_nn_data, &
                    hybrid_nn_data%der_c_mol(:, imol))

            ! scale by reference column to get dI/dc_mol
            hybrid_nn_data%der_c_mol(:, imol) = hybrid_nn_data%der_c_mol(:, imol) * vmol_prior

        END DO

    END SUBROUTINE forward_model_kernel_win_implicit_hybrid_nn

    !-------------------------------------------------------------------------------
    ! SUBROUTINE compute_perturbed_aod_550
    !> @brief Compute aerosol optical depth at 550 nm for perturbed size parameter
    !> @param[in] size_param Perturbed size parameter input to the NN
    !> @param[out] aod_550 Resulting aerosol optical depth at 550 nm
    !-------------------------------------------------------------------------------
    SUBROUTINE compute_perturbed_aod_550(hybrid_nn_data, size_param, aod_550)
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        DOUBLE PRECISION, INTENT(IN) :: size_param
        DOUBLE PRECISION, INTENT(OUT) :: aod_550

        ! TODO: implement this based on the actual scattering/absorption cross-section calculations used in the forward model. 
        ! For now, we will just use a simple placeholder relationship for testing purposes.
        aod_550 = 1.0d0 / size_param  ! Placeholder relationship

    END SUBROUTINE compute_perturbed_aod_550

    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_kernel_win_hybrid_nn_ns_only
    !> @brief Forward model and derivatives for one spectral window using the
    !> neural network code structure, but only executing the non-scattering forward model
    !> @param[in] cabs_mol_fine Fine grid of aerosol absorption coefficients (n_wave_fine)
    !> @param[in] dvmol_prior Prior distribution of vertical profiles (n_wave_fine)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding data for the spectral window
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_kernel_win_hybrid_nn_ns_only( &
                        cabs_mol_fine, &
                        dvmol_prior, &
                        hybrid_nn_data)
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:, :, :), INTENT(IN) :: cabs_mol_fine
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(IN) :: dvmol_prior
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: n_mol
        INTEGER :: imol
        INTEGER :: error_code
        CHARACTER(LEN=256) :: error_message
        !---------------------------------------------------------------------------
 
        n_mol = SIZE(dvmol_prior, 2)  ! number of gases

        ! run non-scattering forward model
        CALL forward_model_kernel_win_ns( &
                            cabs_mol_fine, &
                            dvmol_prior, &
                            hybrid_nn_data, &
                            hybrid_nn_data%ns_der_albedo_fine, &
                            hybrid_nn_data%ns_der_c_mol_fine, &
                            hybrid_nn_data%ns_der_albedo, &
                            hybrid_nn_data%ns_der_c_mol)

        ! Copy over radiances and derivatives
        hybrid_nn_data%spectrum(:) = hybrid_nn_data%ns_spectrum(:)
        hybrid_nn_data%der_albedo = hybrid_nn_data%ns_der_albedo

        ! Then gas columns (loop over each gas)
        DO imol = 1, n_mol
            hybrid_nn_data%der_c_mol(:, imol) = hybrid_nn_data%ns_der_c_mol(:, imol)
        END DO

        ! all aerosol derivatives are zero
        hybrid_nn_data%der_aod = 0.0
        hybrid_nn_data%der_aer_height = 0.0
        hybrid_nn_data%der_size_param = 0.0

    END SUBROUTINE forward_model_kernel_win_hybrid_nn_ns_only 

    !------------------------------------------------------------------------------
    ! SUBROUTINE apply_perturbation
    !> @brief Apply perturbation to hybrid_nn_data nn_inputs_perturbed array
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding data for the spectral window
    !> @param[in] delta_value Step size for finite difference perturbation
    !> @param[in] delta_idx Index of the input parameter to perturb
    !------------------------------------------------------------------------------
    SUBROUTINE apply_perturbation(hybrid_nn_data, &
                        delta_value, &
                        delta_idx)
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        DOUBLE PRECISION, INTENT(IN) :: delta_value
        INTEGER, INTENT(IN) :: delta_idx

        hybrid_nn_data%nn_inputs_perturbed(:) = hybrid_nn_data%nn_inputs(:)
        hybrid_nn_data%nn_inputs_perturbed(delta_idx) = &
                hybrid_nn_data%nn_inputs(delta_idx) + delta_value

    END SUBROUTINE apply_perturbation

    !------------------------------------------------------------------------------
    ! SUBROUTINE finite_difference_win_hybrid_nn
    !> @brief Finite difference approximation of Jacobian for one spectral window 
    !> using the neural network with non-scattering input
    !> @param[in] delta_value Step size for finite difference perturbation
    !> @param[in] delta_idx Index of the input parameter to perturb
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object holding data for the spectral window
    !> @param[in/out] perturbed_hybrid_nn_data Internal object to hold perturbed inputs/outputs
    !> @param[out] derivatives Numerical derivatives (n_wave_meas)
    !------------------------------------------------------------------------------
    SUBROUTINE finite_difference_win_hybrid_nn(delta_value, delta_idx, &
                        hybrid_nn_data, &
                        derivatives)
        ! Input & output parameters
        DOUBLE PRECISION, INTENT(IN) :: delta_value
        INTEGER, INTENT(IN) :: delta_idx
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: derivatives
        !--------------------------------------------------------------------------

        ! Link Fortran arrays to FTorch tensors
        CALL link_tensors_to_perturbed_arrays(hybrid_nn_data)

        ! Run NN for this window --> fills spectrum array in hybrid_nn_data
        CALL forward_model_win_nn_forward_only(hybrid_nn_data)

        ! Compute numerical derivative (forward difference) of NN part only
        derivatives = (hybrid_nn_data%spectrum_perturbed - hybrid_nn_data%spectrum) / delta_value

    END SUBROUTINE finite_difference_win_hybrid_nn


    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_kernel_win_ns
    !> @brief Non-scattering forward model and derivatives for a single spectral window
    !> @param[in] cabs_mol_fine Absorption cross-sections on fine wavelength grid (n_wave_fine x n_layers x n_mol)
    !> @param[in] dvmol_prior Prior vertical column of gases (molecules/m^2) per species (n_layers x n_mol)
    !> @param[in/out] hybrid_nn_data The hybrid_nn_data_type object containing all necessary data
    !> @param[in/out] der_albedo_fine Internal derivative w.r.t. average albedo in the window (n_wave_fine)
    !> @param[in/out] der_c_mol_fine Internal derivative w.r.t. gas column scaling factors (n_wave_fine x n_mol)
    !> @param[out] der_albedo Derivative of non-scattering spectrum w.r.t. average albedo in the window (n_wave_meas)
    !> @param[out] der_c_mol Derivative of non-scattering spectrum w.r.t. gas column scaling factors (n_wave_meas x n_mol)
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_kernel_win_ns( &
                            cabs_mol_fine, &
                            dvmol_prior, &
                            hybrid_nn_data)
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:, :, :), INTENT(IN) :: cabs_mol_fine
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(IN) :: dvmol_prior
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: n_params, n_wave_meas, n_wave_fine, n_isrf, n_mol
        INTEGER :: iwave, is, ie, nidx, i, imol, j
        DOUBLE PRECISION :: ws, we, dw
        DOUBLE PRECISION :: dq
        DOUBLE PRECISION :: resp_linterp, total_resp
        DOUBLE PRECISION :: taua_tot, u
        DOUBLE PRECISION :: max_dw, min_dw
        !--------------------------------------------------------------------------

        n_mol = SIZE(der_c_mol, 2)
        n_wave_meas = hybrid_nn_data%n_wave_meas
        n_wave_fine = hybrid_nn_data%n_wave_fine
        n_isrf = hybrid_nn_data%n_ils_isrf

        u = 1.D0/DBLE(hybrid_nn_data%cos_sza) + 1.D0/DBLE(hybrid_nn_data%cos_vza)
        u = 1.D0/u

        ! calculate hi-res radiance and derivatives
        DO iwave = 1, n_wave_fine
            ! non-scattering radiance
            hybrid_nn_data%ns_spectrum_fine(iwave) = hybrid_nn_data%xbdrf_wave_fine(iwave) * &
                DBLE(hybrid_nn_data%cos_sza)/PI * EXP(-hybrid_nn_data%tau_abs_fine(iwave)/u) * &
                hybrid_nn_data%solar_irrad_fine(iwave)

            ! xbrdf derivative
            hybrid_nn_data%ns_der_albedo_fine(iwave) = &
                hybrid_nn_data%ns_spectrum_fine(iwave) / hybrid_nn_data%xbdrf_wave_fine(iwave)

            ! derivatives w.r.t. gas column scaling factors
            ! assuming fixed vertical profile shape (dvmol_prior)
            DO imol = 1, n_mol
                hybrid_nn_data%ns_der_c_mol_fine(iwave, imol) = -(hybrid_nn_data%ns_spectrum_fine(iwave) / u) * &
                    SUM(cabs_mol_fine(iwave, :, imol) * dvmol_prior(:, imol))
            END DO
        END DO

        ! convolve with ISRF
        dw = hybrid_nn_data%res_fine
        DO iwave = 1, n_wave_meas

            ! get max and min delta wavelength for this channel
            max_dw = hybrid_nn_data%dw_isrf(n_isrf, iwave)
            min_dw = hybrid_nn_data%dw_isrf(1, iwave)

            ! determine wavelength range for convolution
            ws = hybrid_nn_data%wl_meas(iwave) + min_dw
            we = hybrid_nn_data%wl_meas(iwave) + max_dw

            ! determine indices for convolution (making sure they are within bounds)
            is = MAX(1, FLOOR(ABS(ws - hybrid_nn_data%wl_fine(1))/dw))
            ie = MIN(n_wave_fine, CEILING(ABS(we - hybrid_nn_data%wl_fine(1))/dw))
            nidx = ie - is + 1

            ! compute convolved radiance for this channel
            hybrid_nn_data%ns_spectrum(iwave) = 0.D0
            hybrid_nn_data%ns_der_albedo(iwave) = 0.D0
            der_c_mol(iwave, :) = 0.D0
            total_resp = 0.D0
            j = 1
            DO i = 1, nidx
                ! interpolate response from ISRF delta wavelength grid 
                ! to actual delta wavelength

                dq = hybrid_nn_data%wl_fine(is + i - 1) - hybrid_nn_data%wl_meas(iwave)

                ! Advance j until dq is bracketed
                DO WHILE (j < n_isrf)
                    IF (dq > hybrid_nn_data%dw_isrf(j+1, iwave)) THEN
                        j = j + 1
                    ELSE
                        EXIT
                    END IF
                END DO

                ! Linear interpolation
                IF (dq <= hybrid_nn_data%dw_isrf(1, iwave)) THEN
                    resp_linterp = hybrid_nn_data%resp_isrf(1, iwave)
                ELSE IF (dq >= hybrid_nn_data%dw_isrf(n_isrf, iwave)) THEN
                    resp_linterp = hybrid_nn_data%resp_isrf(n_isrf, iwave)
                ELSE
                    resp_linterp = hybrid_nn_data%resp_isrf(j, iwave) + &
                                (hybrid_nn_data%resp_isrf(j+1, iwave) - &
                                hybrid_nn_data%resp_isrf(j, iwave)) * &
                                (dq - hybrid_nn_data%dw_isrf(j, iwave)) / &
                                (hybrid_nn_data%dw_isrf(j+1, iwave) - hybrid_nn_data%dw_isrf(j, iwave))
                END IF

                total_resp = total_resp + resp_linterp
                hybrid_nn_data%ns_spectrum(iwave) = hybrid_nn_data%ns_spectrum(iwave) + &
                        hybrid_nn_data%ns_spectrum_fine(is + i - 1) * resp_linterp

                hybrid_nn_data%ns_der_albedo(iwave) = hybrid_nn_data%ns_der_albedo(iwave) + &
                        hybrid_nn_data%ns_der_albedo_fine(is + i - 1) * resp_linterp
                hybrid_nn_data%ns_der_c_mol(iwave, :) = hybrid_nn_data%ns_der_c_mol(iwave, :) + &
                        hybrid_nn_data%ns_der_c_mol_fine(is + i - 1, :) * resp_linterp
            END DO

            ! normalize by total response
            hybrid_nn_data%ns_spectrum(iwave) = hybrid_nn_data%ns_spectrum(iwave) / total_resp
            hybrid_nn_data%ns_der_albedo(iwave) = hybrid_nn_data%ns_der_albedo(iwave) / total_resp
            hybrid_nn_data%ns_der_c_mol(iwave, :) = hybrid_nn_data%ns_der_c_mol(iwave, :) / total_resp

        END DO

    END SUBROUTINE forward_model_kernel_win_ns

    !------------------------------------------------------------------------------
    !> @brief Links FTorch tensors to perturbed Fortran arrays in the hybrid_nn_data_type
    !> @param[inout] hybrid_nn_data The hybrid_nn_data_type object to link tensors to arrays
    !------------------------------------------------------------------------------
    SUBROUTINE link_tensors_to_perturbed_arrays(hybrid_nn_data)
        TYPE(hybrid_nn_data_type), INTENT(INOUT) :: hybrid_nn_data

        CALL torch_tensor_from_array(hybrid_nn_data%input_tensors(1), &
                                     hybrid_nn_data%nn_inputs_perturbed, &
                                     input_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%proxy_tensors(1), &
                                     hybrid_nn_data%ns_spectrum_perturbed, &
                                     proxy_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%profile_tensors(1), &
                                     hybrid_nn_data%nn_profile_inputs, &
                                     profile_layout, torch_device)
        CALL torch_tensor_from_array(hybrid_nn_data%output_tensors(1), &
                                     hybrid_nn_data%spectrum_perturbed, &
                                     output_layout, torch_device)

    END SUBROUTINE link_tensors_to_perturbed_arrays

END MODULE hybrid_nn_forward_model_jacobian_module
