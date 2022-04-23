!
! Copyright (C) by Argonne National Laboratory
!     See COPYRIGHT in top-level directory
!

subroutine PMPIR_Ineighbor_alltoallw_f08ts(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, &
    rdispls, recvtypes, comm, request, ierror)
    use, intrinsic :: iso_c_binding, only : c_int
    use :: mpi_f08, only : MPI_Datatype, MPI_Comm, MPI_Request
    use :: mpi_f08, only : MPI_ADDRESS_KIND
    use :: mpi_c_interface, only : c_Datatype, c_Comm, c_Request
    use :: mpi_c_interface, only : MPIR_Ineighbor_alltoallw_cdesc, MPIR_Dist_graph_neighbors_count_c

    implicit none

    type(*), dimension(..), intent(in), asynchronous :: sendbuf
    type(*), dimension(..), asynchronous :: recvbuf
    integer, intent(in)  :: sendcounts(*)
    integer, intent(in)  :: recvcounts(*)
    integer(MPI_ADDRESS_KIND), intent(in)  :: sdispls(*)
    integer(MPI_ADDRESS_KIND), intent(in)  :: rdispls(*)
    type(MPI_Datatype), intent(in)  :: sendtypes(*)
    type(MPI_Datatype), intent(in)  :: recvtypes(*)
    type(MPI_Comm), intent(in)  :: comm
    type(MPI_Request), intent(out)  :: request
    integer, optional, intent(out)  :: ierror

    integer(c_int), allocatable :: sendcounts_c(:)
    integer(c_int), allocatable :: recvcounts_c(:)
    integer(MPI_ADDRESS_KIND), allocatable :: sdispls_c(:)
    integer(MPI_ADDRESS_KIND), allocatable :: rdispls_c(:)
    integer(c_Datatype), allocatable :: sendtypes_c(:)
    integer(c_Datatype), allocatable :: recvtypes_c(:)
    integer(c_Comm) :: comm_c
    integer(c_Request) :: request_c
    integer(c_int) :: ierror_c
    integer(c_int) :: err, indegree, outdegree, weighted ! To get length of assumed-size arrays

    comm_c = comm%MPI_VAL
    err = MPIR_Dist_graph_neighbors_count_c(comm_c, indegree, outdegree, weighted)

    if (c_int == kind(0)) then
        ierror_c = MPIR_Ineighbor_alltoallw_cdesc(sendbuf, sendcounts, sdispls, sendtypes(1:outdegree)%MPI_VAL, recvbuf, &
            recvcounts, rdispls, recvtypes(1:indegree)%MPI_VAL, comm%MPI_VAL, request%MPI_VAL)
    else
        sendtypes_c = sendtypes(1:outdegree)%MPI_VAL
        recvtypes_c = recvtypes(1:indegree)%MPI_VAL
        sendcounts_c = sendcounts(1:outdegree)
        recvcounts_c = recvcounts(1:indegree)
        ierror_c = MPIR_Ineighbor_alltoallw_cdesc(sendbuf, sendcounts_c, sdispls, sendtypes_c, recvbuf, recvcounts_c, &
            rdispls, recvtypes_c, comm_c, request_c)
        request%MPI_VAL = request_c
    end if

    if(present(ierror)) ierror = ierror_c

end subroutine PMPIR_Ineighbor_alltoallw_f08ts
