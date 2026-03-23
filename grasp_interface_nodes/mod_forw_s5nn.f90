!------------------------------------------------------------------------------
!  MODULE mod_forw_s5nn
!> @copyright Copyright (c) 2025 SRON, Space Research Organisation Netherlands
!> @author Fiona Lippert (SRON)
!> @brief This module contains the main routines for the S5 neural network
!> forward model.
!------------------------------------------------------------------------------
MODULE mod_forw_s5nn
    USE bridge_grasp_s5nn, ONLY: update_nn_data_from_forw_s5nn, &
                                 update_forw_s5nn_from_nn_data
    USE mod_s5nn_derived_type, ONLY: N_WIN, &
                                     forward_model_characteristics_s5nn, &
                                     forward_model_configuration_s5nn
    IMPLICIT NONE

    !--- Public subroutines implemented in this module
    PUBLIC :: get_s5nn
    PUBLIC :: forw_s5nn

    ! Everything else in this module is private
    PRIVATE

    !------------------------------------------------------------------------------
    CONTAINS

    !------------------------------------------------------------------------------
    !> @brief Main routine to perform S5 neural network forward model calculation
    !------------------------------------------------------------------------------
    SUBROUTINE forw_s5nn(fw_s5nn, fw_cfg_s5nn)
        TYPE(forward_model_characteristics_s5nn), INTENT(INOUT) :: fw_s5nn
        TYPE(forward_model_configuration_s5nn), INTENT(IN) :: fw_cfg_s5nn
        USE mod_alloc_s5nn, ONLY: nn_data
        IMPLICIT NONE
        !------------------------------------------------------------------------------
        INTEGER :: iwin
        !------------------------------------------------------------------------------

        ! Update internal nn_data based on forw_s5nn characteristics
        CALL update_nn_data_from_forw_s5nn(fw_s5nn)

        DO iwin = 1, N_WIN
            ! Check which NN model to use based on configuration
            IF (fw_cfg_s5nn%use_hybrid_model) THEN
                ! Call the hybrid NN forward model for this spectral window
                CALL forward_model_win_nn_hybrid(nn_data%win_data(iwin))
            ELSE
                ! Call the base NN forward model for this spectral window
                CALL forward_model_win_nn_base(nn_data%win_data(iwin))
            END IF
        END DO

        ! Update forw_s5nn characteristics based on internal nn_data
        CALL update_forw_s5nn_from_nn_data(fw_s5nn)

        ! Now, fw_s5nn%radiances contains an array of radiances at measurement wavelengths
        ! for all spectral windows concatenated. If a specific set of measurement
        ! channels is required, we need to extract those here (not implemented yet).

    END SUBROUTINE forw_s5nn

    !------------------------------------------------------------------------------
    !> @brief Routine to load relevant data from disk
    !------------------------------------------------------------------------------
    SUBROUTINE get_s5nn()
        USE mod_alloc_s5nn, ONLY: nn_data
        IMPLICIT NONE
        !------------------------------------------------------------------------------

        ! Load data and NN models for all spectral windows
        CALL initialize_s5nn_data(nn_data)

    END SUBROUTINE get_s5nn

END MODULE mod_forw_s5nn