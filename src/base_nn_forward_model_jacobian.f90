!------------------------------------------------------------------------------
!  MODULE base_nn_forward_model_jacobian_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the subroutines for computing Jacobians of the 
!> S5 base neural network forward model, using finite differences.
!------------------------------------------------------------------------------
MODULE base_nn_forward_model_jacobian_module
    USE ftorch
    USE base_nn_data_type_module
    USE base_nn_forward_model_module
    IMPLICIT NONE
    PRIVATE

    ! Methods implemented in this module
    PUBLIC :: forward_model_kernel_win_nn
    PUBLIC :: finite_difference_win_nn

CONTAINS

    
    !------------------------------------------------------------------------------
    ! SUBROUTINE forward_model_kernel_win_nn
    !> @brief Forward model and derivatives for one spectral window
    !> @param[in] delta_values Array of perturbations for finite difference derivatives (n_nn_inputs)
    !> @param[in] dvmol_prior Prior distribution of vertical profiles (n_wave_fine)
    !> @param[in/out] base_nn_data The base_nn_data_type object holding data for the spectral window
    !> @param[in/out] perturbed_base_nn_data Internal object to hold perturbed inputs/outputs
    !> @param[out] spectrum Output spectrum from forward model (n_wave_meas)
    !> @param[out] der_albedo Derivative of spectrum w.r.t. avg albedo in window (n_wave_meas)
    !> @param[out] der_aod Derivative of spectrum w.r.t. aerosol optical depth (AOD) at 550 nm (n_wave_meas)
    !> @param[out] der_angstrom Derivative of spectrum w.r.t. angstrom exponent (n_wave_meas)
    !> @param[out] der_aer_height Derivative of spectrum w.r.t. aerosol layer height (n_wave_meas)
    !> @param[out] der_c_mol Derivative of spectrum w.r.t. gas column scaling factors (n_wave_meas)
    !------------------------------------------------------------------------------
    SUBROUTINE forward_model_kernel_win_nn( &
                        delta_values, &
                        dvmol_prior, &
                        base_nn_data, &
                        perturbed_base_nn_data, &
                        der_albedo, &
                        der_aod, &
                        der_angstrom, &
                        der_aer_height, &
                        der_c_mol)
        ! Input & output parameters
        DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: delta_values
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(IN) :: dvmol_prior
        TYPE(base_nn_data_type), INTENT(INOUT) :: base_nn_data
        TYPE(base_nn_data_type), INTENT(INOUT) :: perturbed_base_nn_data
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: der_albedo
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: der_aod
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: der_angstrom
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: der_aer_height
        DOUBLE PRECISION, DIMENSION(:, :), INTENT(OUT) :: der_c_mol
        !--------------------------------------------------------------------------
        ! Local variables
        INTEGER :: imol
        INTEGER :: n_mol
        INTEGER, PARAMETER :: n_inputs = 1
        !---------------------------------------------------------------------------
        ! For each input parameter, perturb and compute numerical derivative 
        ! with finite differences

        ! Determine number of gases from size of dvmol_prior
        n_mol = SIZE(der_c_mol, 2)

        ! Run NN forward model with original inputs to get baseline output
        ! --> fills base_nn_data%spectrum
        CALL forward_model_win_nn(base_nn_data)

        ! Now, compute numerical derivatives for each state vector parameter
        ! Start with avg albedo
        CALL finite_difference_win_nn(delta_values(5), 5, &
                base_nn_data, &
                perturbed_base_nn_data, &
                der_albedo)

        ! Then aerosol parameters
        ! (aerosol optical depth, angstrom exponent, layer height)
        CALL finite_difference_win_nn(delta_values(6), 6, &
                base_nn_data, &
                perturbed_base_nn_data, &
                der_aod)
        CALL finite_difference_win_nn(delta_values(7), 7, &
                base_nn_data, &
                perturbed_base_nn_data, &
                der_angstrom)
        CALL finite_difference_win_nn(delta_values(8), 8, &
                base_nn_data, &
                perturbed_base_nn_data, &
                der_aer_height)

        ! Then gas columns (loop over each gas)
        DO imol = 1, n_mol
            CALL finite_difference_win_nn(delta_values(8 + imol), 8 + imol, &
                    base_nn_data, &
                    perturbed_base_nn_data, &
                    der_c_mol(:, imol))
            ! scale by reference column to get dI/dc_mol
            der_c_mol(:, imol) = der_c_mol(:, imol) * SUM(dvmol_prior(:, imol))
        END DO

    END SUBROUTINE forward_model_kernel_win_nn



    !------------------------------------------------------------------------------
    ! SUBROUTINE finite_difference_win_nn
    !> @brief Finite difference approximation of Jacobian for one spectral window 
    !> @param[in] delta_value Step size for finite difference perturbation
    !> @param[in] delta_idx Index of the input parameter to perturb
    !> @param[in/out] base_nn_data The base_nn_data_type object holding reference data
    !> @param[in/out] perturbed_base_nn_data Internal object to hold perturbed inputs/outputs
    !> @param[out] derivatives Numerical derivatives (n_wave_meas)
    !------------------------------------------------------------------------------
    SUBROUTINE finite_difference_win_nn(delta_value, delta_idx, &
                        base_nn_data, &
                        perturbed_base_nn_data, &
                        derivatives)
        ! Input & output parameters
        DOUBLE PRECISION, INTENT(IN) :: delta_value
        INTEGER, INTENT(IN) :: delta_idx
        TYPE(base_nn_data_type), INTENT(INOUT) :: base_nn_data
        TYPE(base_nn_data_type), INTENT(INOUT) :: perturbed_base_nn_data
        DOUBLE PRECISION, DIMENSION(:), INTENT(OUT) :: derivatives
        !------------------------------------------------------------------------------

        ! Perturb input parameter (idx) by a small amount (delta_value)
        perturbed_base_nn_data%nn_inputs(:) = base_nn_data%nn_inputs(:)
        perturbed_base_nn_data%nn_inputs(delta_idx) = &
                perturbed_base_nn_data%nn_inputs(delta_idx) + delta_value

        ! Run NN forward model with perturbed inputs to get perturbed output
        ! --> fills base_nn_data%spectrum
        CALL forward_model_win_nn(perturbed_base_nn_data)

        ! Compute numerical derivative (forward difference) of NN part only
        derivatives = (perturbed_base_nn_data%spectrum - base_nn_data%spectrum) / delta_value

    END SUBROUTINE finite_difference_win_nn

END MODULE base_nn_forward_model_jacobian_module
