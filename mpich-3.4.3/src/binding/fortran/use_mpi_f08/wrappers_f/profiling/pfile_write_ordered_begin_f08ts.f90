!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_File_write_ordered_begin_f08ts(fh, buf, count, datatype, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_File, MPI_Datatype
    use :: mpi_f08, only : MPI_File_f2c, MPI_File_c2f
    use :: mpi_c_interface, only : c_File, c_Datatype
    use :: mpi_c_interface, only : MPIR_File_write_ordered_begin_cdesc

    implicit none

    type(MPI_File), intent(in) :: fh
    type(*), dimension(..), intent(in) :: buf
    integer, intent(in) :: count
    type(MPI_Datatype), intent(in) :: datatype
    integer, optional, intent(out) :: ierror

    integer(c_File) :: fh_c
    integer(c_int) :: count_c
    integer(c_Datatype) :: datatype_c
    integer(c_int) :: ierror_c

    fh_c = MPI_File_f2c(fh%MPI_VAL)
    if (c_int == kind(0)) then
        ierror_c = MPIR_File_write_ordered_begin_cdesc(fh_c, buf, count, datatype%MPI_VAL)
    else
        count_c = count
        datatype_c = datatype%MPI_VAL
        ierror_c = MPIR_File_write_ordered_begin_cdesc(fh_c, buf, count_c, datatype_c)
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_File_write_ordered_begin_f08ts
