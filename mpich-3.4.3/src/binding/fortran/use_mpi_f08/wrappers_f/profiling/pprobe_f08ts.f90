!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Probe_f08(source, tag, comm, status, ierror)
    use, intrinsic :: iso_c_binding, only : c_loc, c_associated
    use, intrinsic :: iso_c_binding, only : c_int, c_ptr
    use :: mpi_f08, only : MPI_Comm, MPI_Status
    use :: mpi_f08, only : MPI_STATUS_IGNORE, MPIR_C_MPI_STATUS_IGNORE, assignment(=)
    use :: mpi_c_interface, only : c_Comm
    use :: mpi_c_interface, only : c_Status
    use :: mpi_c_interface, only : MPIR_Probe_c

    implicit none

    integer, intent(in) :: source
    integer, intent(in) :: tag
    type(MPI_Comm), intent(in) :: comm
    type(MPI_Status), target :: status
    integer, optional, intent(out) :: ierror

    integer(c_int) :: source_c
    integer(c_int) :: tag_c
    integer(c_Comm) :: comm_c
    type(c_Status), target :: status_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        if (c_associated(c_loc(status), c_loc(MPI_STATUS_IGNORE))) then
            ierror_c = MPIR_Probe_c(source, tag, comm%MPI_VAL, MPIR_C_MPI_STATUS_IGNORE)
        else
            ierror_c = MPIR_Probe_c(source, tag, comm%MPI_VAL, c_loc(status))
        end if
    else
        source_c = source
        tag_c = tag
        comm_c = comm%MPI_VAL
        if (c_associated(c_loc(status), c_loc(MPI_STATUS_IGNORE))) then
            ierror_c = MPIR_Probe_c(source_c, tag_c, comm_c, MPIR_C_MPI_STATUS_IGNORE)
        else
            ierror_c = MPIR_Probe_c(source_c, tag_c, comm_c, c_loc(status_c))
            status = status_c
        end if
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Probe_f08
