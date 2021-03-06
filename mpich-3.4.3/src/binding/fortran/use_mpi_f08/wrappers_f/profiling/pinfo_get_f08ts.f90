!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Info_get_f08(info, key, valuelen, value, flag, ierror)
    use, intrinsic :: iso_c_binding, only : c_int, c_char
    use :: mpi_f08, only : MPI_Info
    use :: mpi_c_interface, only : c_Info
    use :: mpi_c_interface, only : MPIR_Info_get_c
    use :: mpi_c_interface, only : MPIR_Fortran_string_c2f
    use :: mpi_c_interface, only : MPIR_Fortran_string_f2c

    implicit none

    type(MPI_Info), intent(in) :: info
    character(len=*), intent(in) :: key
    integer, intent(in) :: valuelen
    character(len=valuelen), intent(out) :: value
    logical, intent(out) :: flag
    integer, optional, intent(out) :: ierror

    integer(c_Info) :: info_c
    character(kind=c_char) :: key_c(len_trim(key)+1)
    integer(c_int) :: valuelen_c
    character(kind=c_char) :: value_c(valuelen+1)
    integer(c_int) :: flag_c
    integer(c_int) :: ierror_c

    call MPIR_Fortran_string_f2c(key, key_c)

    if (c_int == kind(0)) then
        ierror_c = MPIR_Info_get_c(info%MPI_VAL, key_c, valuelen, value_c, flag_c)
    else
        info_c = info%MPI_VAL
        valuelen_c = valuelen
        ierror_c = MPIR_Info_get_c(info_c, key_c, valuelen_c, value_c, flag_c)
    end if

    flag = (flag_c /= 0)

    if (flag) then  ! value is unchanged when flag is false
        call MPIR_Fortran_string_c2f(value_c, value)
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Info_get_f08
