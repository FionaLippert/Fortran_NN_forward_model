!------------------------------------------------------------------------------
!  MODULE hybrid_nn_data_type_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module defines the data type holding relevant data for the hybrid
!> neural network forward model, including inputs and outputs for non-scattering
!> calculations and neural networks
!------------------------------------------------------------------------------
MODULE hybrid_nn_data_type_module
    USE ftorch
    USE, INTRINSIC :: iso_fortran_env, only: wp => real32
    IMPLICIT NONE
    PRIVATE

    ! methods implemented in this module
    PUBLIC :: allocate_hybrid_nn_data
    PUBLIC :: deallocate_hybrid_nn_data

    !------------------------------------------------------------------------------

    TYPE, PUBLIC :: hybrid_nn_data_type

        ! the traced PyTorch neural network, loaded via FTorch
        TYPE(torch_model) :: nn_model

        ! inputs and outputs for the neural network
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: nn_inputs                ! input array for the neural network (n_nn_inputs)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_spectrum              ! non-scattering spectrum array (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: nn_profile_inputs     ! input array for profile variables, e.g. temperature (n_layers, n_variables)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: spectrum                 ! final spectrum (n_wave_meas)
        TYPE(torch_tensor), DIMENSION(1) :: input_tensors                       ! first input tensor holding input array (n_nn_inputs)
        TYPE(torch_tensor), DIMENSION(1) :: proxy_tensors                       ! second input tensor holding non-scattering array (n_wave_meas)
        TYPE(torch_tensor), DIMENSION(1) :: profile_tensors                 ! third input tensor holding profile input arrays (n_nn_profile_inputs, n_layers)
        TYPE(torch_tensor), DIMENSION(1) :: output_tensors                      ! output tensor holding final spectrum (n_wave_meas)

        ! perturbed inputs and outputs (for finite difference calculations)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: nn_inputs_perturbed  ! perturbed input array for the neural network (n_nn_inputs)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: spectrum_perturbed   ! perturbed final spectrum (n_wave_meas)

        ! NN derivatives
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: der_albedo           ! derivative of spectrum w.r.t. albedo (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: der_aod              ! derivative of spectrum w.r.t. aerosol optical depth (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: der_size_param       ! derivative of spectrum w.r.t. aerosol size parameter (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: der_aer_height       ! derivative of spectrum w.r.t. aerosol height (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: der_c_mol         ! derivative of spectrum w.r.t. column density (n_wave_meas, n_gases)
        
        ! non-scattering derivatives
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_der_albedo        ! derivative of non-scattering spectrum w.r.t. albedo (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: ns_der_c_mol      ! derivative of non-scattering spectrum w.r.t. column density (n_wave_meas, n_gases)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_der_albedo_fine   ! derivative of non-scattering spectrum on fine grid w.r.t. albedo (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:, :), ALLOCATABLE :: ns_der_c_mol_fine ! derivative of non-scattering spectrum on fine grid w.r.t. column density (n_wave_fine, n_gases)

        ! data needed for measurement channel matching
        DOUBLE PRECISION, ALLOCATABLE :: wl_meas(:)                         ! measurement wavelengths in nm (n_wave_meas)
        DOUBLE PRECISION :: res_meas                                        ! measurement resolution in nm (assumed constant)
        INTEGER :: n_wave_meas                                              ! number of measurement wavelengths

        ! data needed for or filled by non-scattering calculations
        DOUBLE PRECISION :: cos_sza                                         ! cosine of solar zenith angle
        DOUBLE PRECISION :: cos_vza                                         ! cosine of viewing zenith angle
        DOUBLE PRECISION, ALLOCATABLE :: wl_fine(:)                         ! fine wavelength grid in nm (n_wave_fine)
        DOUBLE PRECISION, ALLOCATABLE :: resp_isrf(:, :)                    ! ISRF response function (n_ils_isrf, n_wave_meas)
        DOUBLE PRECISION, ALLOCATABLE :: dw_isrf(:, :)                      ! ISRF delta wavelength grid (n_ils_isrf, n_wave_meas)
        DOUBLE PRECISION, ALLOCATABLE :: solar_irrad_fine(:)                ! solar irradiance (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: xbdrf_wave_fine      ! surface BRDF on fine grid (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: tau_abs_fine         ! absorption optical depth on fine grid (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_spectrum_fine     ! non-scattering spectrum on fine grid (n_wave_fine)
        INTEGER :: n_ils_isrf                                               ! number of ISRF points
        INTEGER :: n_wave_fine                                              ! number of fine wavelengths
        DOUBLE PRECISION :: res_fine                                        ! fine wavelength grid resolution (nm)

        ! perturbed inputs for implicit hybrid NN derivatives
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: ns_spectrum_perturbed ! perturbed non-scattering spectrum (n_wave_meas)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: tau_abs_fine_perturbed ! perturbed absorption optical depth on fine grid (n_wave_fine)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: xbdrf_wave_fine_perturbed     ! perturbed surface BRDF on fine grid (n_wave_fine)

        ! TODO: add additional data fields if needed, e.g. for computing AOD

    END TYPE hybrid_nn_data_type

CONTAINS
  
    !------------------------------------------------------------------------------
    !> @brief Allocates memory for the hybrid_nn_data_type
    !> @param[inout] this The hybrid_nn_data_type object to allocate memory for
    !> @param[in] n_inputs Number of NN input features
    !> @param[in] n_layers Number of atmospheric layers (for profile inputs)
    !> @param[in] n_profile_vars Number of profile variables (e.g. temperature, pressure, etc.)
    !> @param[in] n_wave_meas Number of measurement wavelengths
    !> @param[in] n_wave_fine Number of fine wavelengths
    !> @param[in] n_ils_isrf Number of ISRF points
    !------------------------------------------------------------------------------
    SUBROUTINE allocate_hybrid_nn_data(this, n_inputs, n_layers, n_profile_vars, &
                                        n_gases, n_wave_meas, n_wave_fine, n_ils_isrf)
        CLASS(hybrid_nn_data_type), INTENT(INOUT) :: this
        INTEGER, INTENT(IN) :: n_inputs, n_layers, n_profile_vars, n_gases
        INTEGER, INTENT(IN) :: n_wave_meas, n_wave_fine, n_ils_isrf

        ALLOCATE(this%nn_inputs(n_inputs))
        ALLOCATE(this%ns_spectrum(n_wave_meas))
        ALLOCATE(this%nn_profile_inputs(n_layers, n_profile_vars))
        ALLOCATE(this%spectrum(n_wave_meas))

        ALLOCATE(this%der_albedo(n_wave_meas))
        ALLOCATE(this%der_aod(n_wave_meas))
        ALLOCATE(this%der_size_param(n_wave_meas))
        ALLOCATE(this%der_aer_height(n_wave_meas))
        ALLOCATE(this%der_c_mol(n_wave_meas, n_gases))

        ALLOCATE(this%ns_der_albedo(n_wave_meas))
        ALLOCATE(this%ns_der_c_mol(n_wave_meas, n_gases))
        ALLOCATE(this%ns_der_albedo_fine(n_wave_fine))
        ALLOCATE(this%ns_der_c_mol_fine(n_wave_fine, n_gases))

        ALLOCATE(this%wl_meas(n_wave_meas))
        ALLOCATE(this%wl_fine(n_wave_fine))
        ALLOCATE(this%resp_isrf(n_ils_isrf, n_wave_meas))
        ALLOCATE(this%dw_isrf(n_ils_isrf, n_wave_meas))
        ALLOCATE(this%solar_irrad_fine(n_wave_fine))
        ALLOCATE(this%xbdrf_wave_fine(n_wave_fine))
        ALLOCATE(this%tau_abs_fine(n_wave_fine))
        ALLOCATE(this%ns_spectrum_fine(n_wave_fine))

        ALLOCATE(this%nn_inputs_perturbed(n_inputs))
        ALLOCATE(this%spectrum_perturbed(n_wave_meas))
        ALLOCATE(this%ns_spectrum_perturbed(n_wave_meas))
        ALLOCATE(this%tau_abs_fine_perturbed(n_wave_fine))
        ALLOCATE(this%xbdrf_wave_fine_perturbed(n_wave_fine))

    END SUBROUTINE allocate_hybrid_nn_data

    !------------------------------------------------------------------------------
    !> @brief Deallocates memory for the hybrid_nn_data_type
    !> @param[inout] this The hybrid_nn_data_type object to deallocate memory for
    !------------------------------------------------------------------------------
    SUBROUTINE deallocate_hybrid_nn_data(this)
        CLASS(hybrid_nn_data_type), INTENT(INOUT) :: this

        IF (ALLOCATED(this%nn_inputs)) DEALLOCATE(this%nn_inputs)
        IF (ALLOCATED(this%ns_spectrum)) DEALLOCATE(this%ns_spectrum)
        IF (ALLOCATED(this%nn_profile_inputs)) DEALLOCATE(this%nn_profile_inputs)
        IF (ALLOCATED(this%spectrum)) DEALLOCATE(this%spectrum)

        IF (ALLOCATED(this%der_albedo)) DEALLOCATE(this%der_albedo)
        IF (ALLOCATED(this%der_aod)) DEALLOCATE(this%der_aod)
        IF (ALLOCATED(this%der_size_param)) DEALLOCATE(this%der_size_param)
        IF (ALLOCATED(this%der_aer_height)) DEALLOCATE(this%der_aer_height)
        IF (ALLOCATED(this%der_c_mol)) DEALLOCATE(this%der_c_mol)

        IF (ALLOCATED(this%ns_der_albedo)) DEALLOCATE(this%ns_der_albedo)
        IF (ALLOCATED(this%ns_der_c_mol)) DEALLOCATE(this%ns_der_c_mol)
        IF (ALLOCATED(this%ns_der_albedo_fine)) DEALLOCATE(this%ns_der_albedo_fine)
        IF (ALLOCATED(this%ns_der_c_mol_fine)) DEALLOCATE(this%ns_der_c_mol_fine)

        IF (ALLOCATED(this%wl_meas)) DEALLOCATE(this%wl_meas)
        IF (ALLOCATED(this%wl_fine)) DEALLOCATE(this%wl_fine)
        IF (ALLOCATED(this%resp_isrf)) DEALLOCATE(this%resp_isrf)
        IF (ALLOCATED(this%dw_isrf)) DEALLOCATE(this%dw_isrf)
        IF (ALLOCATED(this%solar_irrad_fine)) DEALLOCATE(this%solar_irrad_fine)
        IF (ALLOCATED(this%xbdrf_wave_fine)) DEALLOCATE(this%xbdrf_wave_fine)
        IF (ALLOCATED(this%tau_abs_fine)) DEALLOCATE(this%tau_abs_fine)
        IF (ALLOCATED(this%ns_spectrum_fine)) DEALLOCATE(this%ns_spectrum_fine)

        IF (ALLOCATED(this%nn_inputs_perturbed)) DEALLOCATE(this%nn_inputs_perturbed)
        IF (ALLOCATED(this%spectrum_perturbed)) DEALLOCATE(this%spectrum_perturbed)
        IF (ALLOCATED(this%ns_spectrum_perturbed)) DEALLOCATE(this%ns_spectrum_perturbed)
        IF (ALLOCATED(this%tau_abs_fine_perturbed)) DEALLOCATE(this%tau_abs_fine_perturbed)
        IF (ALLOCATED(this%xbdrf_wave_fine_perturbed)) DEALLOCATE(this%xbdrf_wave_fine_perturbed)

        ! delete FTorch tensors and model
        CALL torch_delete(this%input_tensors(1))
        CALL torch_delete(this%proxy_tensors(1))
        CALL torch_delete(this%profile_tensors(1))
        CALL torch_delete(this%output_tensors(1))
        CALL torch_delete(this%nn_model)

    END SUBROUTINE deallocate_hybrid_nn_data

END MODULE hybrid_nn_data_type_module