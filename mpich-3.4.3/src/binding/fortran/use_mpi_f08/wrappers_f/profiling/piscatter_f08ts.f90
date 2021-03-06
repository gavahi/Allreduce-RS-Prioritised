!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Iscatter_f08ts(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, &
    root, comm, request, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Datatype, MPI_Comm, MPI_Request
    use :: mpi_c_interface, only : c_Datatype, c_Comm, c_Request
    use :: mpi_c_interface, only : MPIR_Iscatter_cdesc

    implicit none

    type(*), dimension(..), intent(in), asynchronous :: sendbuf
    type(*), dimension(..), asynchronous :: recvbuf
    integer, intent(in) :: sendcount
    integer, intent(in) :: recvcount
    integer, intent(in) :: root
    type(MPI_Datatype), intent(in) :: sendtype
    type(MPI_Datatype), intent(in) :: recvtype
    type(MPI_Comm), intent(in) :: comm
    type(MPI_Request), intent(out) :: request
    integer, optional, intent(out) :: ierror

    integer(c_int) :: sendcount_c
    integer(c_int) :: recvcount_c
    integer(c_int) :: root_c
    integer(c_Datatype) :: sendtype_c
    integer(c_Datatype) :: recvtype_c
    integer(c_Comm) :: comm_c
    integer(c_Request) :: request_c
    integer(c_int) :: ierror_c

    if (c_int == kind(0)) then
        ierror_c = MPIR_Iscatter_cdesc(sendbuf, sendcount, sendtype%MPI_VAL, recvbuf, recvcount, recvtype%MPI_VAL, &
            root, comm%MPI_VAL, request%MPI_VAL)
    else
        sendcount_c = sendcount
        sendtype_c = sendtype%MPI_VAL
        recvcount_c = recvcount
        recvtype_c = recvtype%MPI_VAL
        root_c = root
        comm_c = comm%MPI_VAL
        ierror_c = MPIR_Iscatter_cdesc(sendbuf, sendcount_c, sendtype_c, recvbuf, recvcount_c, recvtype_c, &
            root_c, comm_c, request_c)
        request%MPI_VAL = request_c
    end if

    if (present(ierror)) ierror = ierror_c

end subroutine PMPIR_Iscatter_f08ts
