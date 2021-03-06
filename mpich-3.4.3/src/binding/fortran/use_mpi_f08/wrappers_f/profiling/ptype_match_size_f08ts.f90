!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Type_match_size_f08(typeclass, size, datatype, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Datatype
    use :: mpi_c_interface, only : c_Datatype
    use :: mpi_c_interface, only : MPIR_Type_match_size_c

    implicit none

    integer, intent(in) :: typeclass
    integer, intent(in) :: size
    type(MPI_Datatype), intent(out) :: datatype
    integer, optional, intent(out) :: ierror

    integer(c_int) :: typeclass_c
    integer(c_int) :: size_c
    integer(c_Datatype) :: datatype_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        ierror_c = MPIR_Type_match_size_c(typeclass, size, datatype%MPI_VAL)
    else
        typeclass_c = typeclass
        size_c = size
        ierror_c = MPIR_Type_match_size_c(typeclass_c, size_c, datatype_c)
        datatype%MPI_VAL = datatype_c
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Type_match_size_f08
