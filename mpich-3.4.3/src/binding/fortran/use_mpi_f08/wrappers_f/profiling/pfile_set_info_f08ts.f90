!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_File_set_info_f08(fh, info, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_File, MPI_Info
    use :: mpi_f08, only : MPI_File_f2c, MPI_File_c2f
    use :: mpi_c_interface, only : c_File, c_Info
    use :: mpi_c_interface, only : MPIR_File_set_info_c

    implicit none

    type(MPI_File), intent(in) :: fh
    type(MPI_Info), intent(in) :: info
    integer, optional, intent(out) :: ierror

    integer(c_File) :: fh_c
    integer(c_Info) :: info_c
    integer(c_int) :: ierror_c

    fh_c = MPI_File_f2c(fh%MPI_VAL)
    if (c_int == kind(0)) then
        ierror_c = MPIR_File_set_info_c(fh_c, info%MPI_VAL)
    else
        info_c = info%MPI_VAL
        ierror_c = MPIR_File_set_info_c(fh_c, info_c)
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_File_set_info_f08
