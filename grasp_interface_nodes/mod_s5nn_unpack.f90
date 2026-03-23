!------------------------------------------------------------------------------
!  MODULE mod_s5nn_unpack
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the unpack routines for the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE mod_s5nn_unpack
    IMPLICIT NONE

    !--- Public subroutines implemented in this module
    PUBLIC :: unpack_forward_model_s5nn

    ! Everything else in this module is private
    PRIVATE
 
    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Unpacks relevant characteristics from the state vector 
    !------------------------------------------------------------------------------
    SUBROUTINE unpack_forward_model_s5nn(RIN, ipix, APSING, forw_s5nn, par_type_val)
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        TYPE(retr_input_settings), INTENT(IN) :: RIN
        INTEGER, INTENT(IN) :: ipix
        REAL, DIMENSION(KPARS), INTENT(IN) :: APSING
        TYPE(forward_model_characteristics_s5nn), INTENT(INOUT) :: forw_s5nn
        LOGICAL, DIMENSION(KIDIM1), INTENT(INOUT) :: par_type_val
        !--------------------------------------------------------------------------

        ! TODO: Unpack relevant characteristics from the state vector

    END SUBROUTINE unpack_forward_model_s5nn

END MODULE mod_s5nn_unpack