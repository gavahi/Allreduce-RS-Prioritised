!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Request_get_status_f08(request, flag, status, ierror)
    use, intrinsic :: iso_c_binding, only : c_loc, c_associated
    use, intrinsic :: iso_c_binding, only : c_int, c_ptr
    use :: mpi_f08, only : MPI_Request, MPI_Status
    use :: mpi_f08, only : MPI_STATUS_IGNORE, MPIR_C_MPI_STATUS_IGNORE, assignment(=)
    use :: mpi_c_interface, only : c_Request
    use :: mpi_c_interface, only : c_Status
    use :: mpi_c_interface, only : MPIR_Request_get_status_c

    implicit none

    type(MPI_Request), intent(in) :: request
    logical, intent(out) :: flag
    type(MPI_Status), target :: status
    integer, optional, intent(out) :: ierror

    integer(c_Request) :: request_c
    integer(c_int) :: flag_c
    type(c_Status), target :: status_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        if (c_associated(c_loc(status), c_loc(MPI_STATUS_IGNORE))) then
            ierror_c = MPIR_Request_get_status_c(request%MPI_VAL, flag_c, MPIR_C_MPI_STATUS_IGNORE)
        else
            ierror_c = MPIR_Request_get_status_c(request%MPI_VAL, flag_c, c_loc(status))
        end if
    else
        request_c = request%MPI_VAL
        if (c_associated(c_loc(status), c_loc(MPI_STATUS_IGNORE))) then
            ierror_c = MPIR_Request_get_status_c(request_c, flag_c, MPIR_C_MPI_STATUS_IGNORE)
        else
            ierror_c = MPIR_Request_get_status_c(request_c, flag_c, c_loc(status_c))
            status = status_c
        end if
    end if

    flag = (flag_c /= 0)
    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Request_get_status_f08
