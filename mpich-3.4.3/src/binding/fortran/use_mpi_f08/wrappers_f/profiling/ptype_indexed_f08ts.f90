!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Type_indexed_f08(count, array_of_blocklengths, &
    array_of_displacements, oldtype, newtype, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Datatype
    use :: mpi_c_interface, only : c_Datatype
    use :: mpi_c_interface, only : MPIR_Type_indexed_c

    implicit none

    integer, intent(in) :: count
    integer, intent(in) :: array_of_blocklengths(count)
    integer, intent(in) :: array_of_displacements(count)
    type(MPI_Datatype), intent(in) :: oldtype
    type(MPI_Datatype), intent(out) :: newtype
    integer, optional, intent(out) :: ierror

    integer(c_int) :: count_c
    integer(c_int) :: array_of_blocklengths_c(count)
    integer(c_int) :: array_of_displacements_c(count)
    integer(c_Datatype) :: oldtype_c
    integer(c_Datatype) :: newtype_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        ierror_c = MPIR_Type_indexed_c(count, array_of_blocklengths, array_of_displacements, oldtype%MPI_VAL, &
            newtype%MPI_VAL)
    else
        count_c = count
        array_of_blocklengths_c = array_of_blocklengths
        array_of_displacements_c = array_of_displacements
        oldtype_c = oldtype%MPI_VAL
        ierror_c = MPIR_Type_indexed_c(count_c, array_of_blocklengths_c, array_of_displacements_c, &
            oldtype_c, newtype_c)
        newtype%MPI_VAL = newtype_c
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Type_indexed_f08
