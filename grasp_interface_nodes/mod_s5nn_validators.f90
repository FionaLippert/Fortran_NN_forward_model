!------------------------------------------------------------------------------
!  MODULE mod_s5nn_validators
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains validation subroutines for the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE mod_s5nn_validators
    ! TODO: where is retr_input_settings defined?
    ! TODO: where is segment_data defined?
    ! TODO: where is meas_present_status defined?
    IMPLICIT NONE

    ! Public subroutines implemented in this module
    PUBLIC :: validator_s5nn_parameters

    ! Everything else in this module is private
    PRIVATE

    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Validates the S5 neural network forward model
    !> @param[in] RIN The GRASP input configuration data structure
    !> @param[in] segment The segment data structure
    !> @param[in] meas_status The measurement status structure
    !------------------------------------------------------------------------------
    SUBROUTINE validator_s5nn_parameters(RIN, segment, meas_status)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        ! Inputs & outputs
        TYPE(retr_input_settings), INTENT(IN) :: RIN
        TYPE(segment_data), INTENT(IN) :: segment
        TYPE(meas_present_status), INTENT(IN) :: meas_status
        !------------------------------------------------------------------------------

        ! TODO: implement validation checks for S5NN parameters and throw errors if needed
        ! Errors can be thrown using the GRASP error handling system: 
        ! write(tmp_message,'(a)') &
        ! 'Settings inconsistency: error info...'
        ! G_ERROR(trim(tmp_message))

        ! 1. Make sure that measurement channels are within NIR, SWIR1, SWIR3

        ! 2. Make sure that aerosol model is single mode powerlaw model

        ! 3. Make sure that gases match the gases used for NN training


    END SUBROUTINE validator_s5nn_parameters

END MODULE mod_s5nn_validators