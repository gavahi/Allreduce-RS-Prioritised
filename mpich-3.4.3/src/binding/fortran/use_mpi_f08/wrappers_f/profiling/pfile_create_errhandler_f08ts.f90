!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_File_create_errhandler_f08(file_errhandler_fn, errhandler, ierror)
    use, intrinsic :: iso_c_binding, only : c_funloc
    use, intrinsic :: iso_c_binding, only : c_int, c_funptr
    use :: mpi_f08, only : MPI_Errhandler
    use :: mpi_f08, only : MPI_File_errhandler_function
    use :: mpi_c_interface, only : c_Errhandler
    use :: mpi_c_interface, only : MPIR_File_create_errhandler_c

    implicit none

    procedure(MPI_File_errhandler_function) :: file_errhandler_fn
    type(MPI_Errhandler), intent(out) :: errhandler
    integer, optional, intent(out) :: ierror

    type(c_funptr) :: file_errhandler_fn_c
    integer(c_Errhandler) :: errhandler_c
    integer(c_int) :: ierror_c

    file_errhandler_fn_c = c_funloc(file_errhandler_fn)
    if (c_int == kind(0)) then
        ierror_c = MPIR_File_create_errhandler_c(file_errhandler_fn_c, errhandler%MPI_VAL)
    else
        ierror_c = MPIR_File_create_errhandler_c(file_errhandler_fn_c, errhandler_c)
        errhandler%MPI_VAL = errhandler_c
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_File_create_errhandler_f08
