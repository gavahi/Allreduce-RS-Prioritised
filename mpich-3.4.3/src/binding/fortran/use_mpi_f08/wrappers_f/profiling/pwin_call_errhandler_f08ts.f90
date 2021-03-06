!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Win_call_errhandler_f08(win, errorcode, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Win
    use :: mpi_c_interface, only : c_Win
    use :: mpi_c_interface, only : MPIR_Win_call_errhandler_c

    implicit none

    type(MPI_Win), intent(in) :: win
    integer, intent(in) :: errorcode
    integer, optional, intent(out) :: ierror

    integer(c_Win) :: win_c
    integer(c_int) :: errorcode_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        ierror_c = MPIR_Win_call_errhandler_c(win%MPI_VAL, errorcode)
    else
        win_c = win%MPI_VAL
        errorcode_c = errorcode
        ierror_c = MPIR_Win_call_errhandler_c(win_c, errorcode_c)
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Win_call_errhandler_f08
