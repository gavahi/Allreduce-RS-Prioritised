!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

function PMPIR_Aint_diff_f08(addr1, addr2) result(res)
    use :: mpi_f08_compile_constants, only : MPI_ADDRESS_KIND
    use :: mpi_c_interface_nobuf, only : MPIR_Aint_diff_c
    implicit none
    integer(MPI_ADDRESS_KIND), intent(in) :: addr1, addr2
    integer(MPI_ADDRESS_KIND) :: res

    res = MPIR_Aint_diff_c(addr1, addr2)
end function PMPIR_Aint_diff_f08
