!------------------------------------------------------------------------------
!  MODULE base_nn_data_type_module
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module defines the data type holding relevant data for the base
!> neural network forward model, including inputs, outputs, and the neural network
!------------------------------------------------------------------------------
MODULE base_nn_data_type_module
    USE ftorch
    IMPLICIT NONE
    PRIVATE

    ! methods implemented in this module
    PUBLIC :: allocate_base_nn_data
    PUBLIC :: deallocate_base_nn_data

    !------------------------------------------------------------------------------

    TYPE, PUBLIC :: base_nn_data_type

        ! the traced PyTorch neural network, loaded via FTorch
        TYPE(torch_model) :: nn_model

        ! inputs and outputs for the neural network
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: nn_inputs           ! input array for the neural network (n_nn_inputs)
        DOUBLE PRECISION, DIMENSION(:), ALLOCATABLE :: spectrum            ! final spectrum (n_wave_meas)
        TYPE(torch_tensor), DIMENSION(1) :: input_tensors                  ! first input tensor, holding input array (n_nn_inputs)
        TYPE(torch_tensor), DIMENSION(1) :: output_tensors                 ! output tensor, holding final spectrum (n_wave_meas)

        ! data needed for measurement channel matching
        DOUBLE PRECISION, ALLOCATABLE :: wl_meas(:)                         ! measurement wavelengths in nm (n_wave_meas)
        DOUBLE PRECISION :: res_meas                                        ! measurement resolution in nm (assumed constant)
        INTEGER :: n_wave_meas                                              ! number of measurement wavelengths

    END TYPE base_nn_data_type

CONTAINS
  
    !------------------------------------------------------------------------------
    !> @brief Allocates memory for the base_nn_data_type
    !> @param[inout] this The base_nn_data_type object to allocate memory for
    !> @param[in] n_inputs Number of NN input features
    !> @param[in] n_wave_meas Number of measurement wavelengths
    !------------------------------------------------------------------------------
    SUBROUTINE allocate_base_nn_data(this, n_inputs, n_wave_meas)
        CLASS(base_nn_data_type), INTENT(INOUT) :: this
        INTEGER, INTENT(IN) :: n_inputs, n_wave_meas

        ALLOCATE(this%nn_inputs(n_inputs))
        ALLOCATE(this%spectrum(n_wave_meas))
        ALLOCATE(this%wl_meas(n_wave_meas))

    END SUBROUTINE allocate_base_nn_data

    !------------------------------------------------------------------------------
    !> @brief Deallocates memory for the base_nn_data_type
    !> @param[inout] this The base_nn_data_type object to deallocate memory for
    !------------------------------------------------------------------------------
    SUBROUTINE deallocate_base_nn_data(this)
        CLASS(base_nn_data_type), INTENT(INOUT) :: this

        IF (ALLOCATED(this%nn_inputs)) DEALLOCATE(this%nn_inputs)
        IF (ALLOCATED(this%spectrum)) DEALLOCATE(this%spectrum)
        IF (ALLOCATED(this%wl_meas)) DEALLOCATE(this%wl_meas)

        ! delete FTorch tensors and model
        CALL torch_delete(this%input_tensors(1))
        CALL torch_delete(this%output_tensors(1))
        CALL torch_delete(this%nn_model)

    END SUBROUTINE deallocate_base_nn_data

END MODULE base_nn_data_type_module