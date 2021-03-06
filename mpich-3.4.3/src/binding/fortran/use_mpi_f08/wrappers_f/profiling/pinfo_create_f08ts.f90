!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Info_create_f08(info, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Info
    use :: mpi_c_interface, only : c_Info
    use :: mpi_c_interface, only : MPIR_Info_create_c

    implicit none

    type(MPI_Info), intent(out) :: info
    integer, optional, intent(out) :: ierror

    integer(c_Info) :: info_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        ierror_c = MPIR_Info_create_c(info%MPI_VAL)
    else
        ierror_c = MPIR_Info_create_c(info_c)
        info%MPI_VAL = info_c
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Info_create_f08
