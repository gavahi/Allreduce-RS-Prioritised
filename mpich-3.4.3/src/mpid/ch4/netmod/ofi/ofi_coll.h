/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef OFI_COLL_H_INCLUDED
#define OFI_COLL_H_INCLUDED

#include "ofi_impl.h"
#include "ch4_csel_container.h"

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_barrier(MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_BARRIER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_BARRIER);

    mpi_errno = MPIR_Barrier_impl(comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_BARRIER);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_bcast(void *buffer, int count, MPI_Datatype datatype,
                                                int root, MPIR_Comm * comm,
                                                MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_BCAST);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_BCAST);

    mpi_errno = MPIR_Bcast_impl(buffer, count, datatype, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_BCAST);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_allreduce(const void *sendbuf, void *recvbuf, int count,
                                                    MPI_Datatype datatype, MPI_Op op,
                                                    MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLREDUCE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLREDUCE);

    mpi_errno = MPIR_Allreduce_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLREDUCE);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_allgather(const void *sendbuf, int sendcount,
                                                    MPI_Datatype sendtype, void *recvbuf,
                                                    int recvcount, MPI_Datatype recvtype,
                                                    MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLGATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLGATHER);

    mpi_errno = MPIR_Allgather_impl(sendbuf, sendcount, sendtype,
                                    recvbuf, recvcount, recvtype, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLGATHER);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_allgatherv(const void *sendbuf, int sendcount,
                                                     MPI_Datatype sendtype, void *recvbuf,
                                                     const int *recvcounts, const int *displs,
                                                     MPI_Datatype recvtype, MPIR_Comm * comm,
                                                     MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLGATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLGATHERV);

    mpi_errno = MPIR_Allgatherv_impl(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcounts, displs, recvtype, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLGATHERV);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_gather(const void *sendbuf, int sendcount,
                                                 MPI_Datatype sendtype, void *recvbuf,
                                                 int recvcount, MPI_Datatype recvtype, int root,
                                                 MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_GATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_GATHER);

    mpi_errno = MPIR_Gather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                 recvcount, recvtype, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_GATHER);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_gatherv(const void *sendbuf, int sendcount,
                                                  MPI_Datatype sendtype, void *recvbuf,
                                                  const int *recvcounts, const int *displs,
                                                  MPI_Datatype recvtype, int root, MPIR_Comm * comm,
                                                  MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_GATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_GATHERV);

    mpi_errno = MPIR_Gatherv_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcounts, displs, recvtype, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_GATHERV);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;

}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_scatter(const void *sendbuf, int sendcount,
                                                  MPI_Datatype sendtype, void *recvbuf,
                                                  int recvcount, MPI_Datatype recvtype, int root,
                                                  MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_SCATTER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_SCATTER);

    mpi_errno = MPIR_Scatter_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_SCATTER);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_scatterv(const void *sendbuf, const int *sendcounts,
                                                   const int *displs, MPI_Datatype sendtype,
                                                   void *recvbuf, int recvcount,
                                                   MPI_Datatype recvtype, int root,
                                                   MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_SCATTERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_SCATTERV);

    mpi_errno = MPIR_Scatterv_impl(sendbuf, sendcounts, displs, sendtype, recvbuf,
                                   recvcount, recvtype, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_SCATTERV);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_alltoall(const void *sendbuf, int sendcount,
                                                   MPI_Datatype sendtype, void *recvbuf,
                                                   int recvcount, MPI_Datatype recvtype,
                                                   MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLTOALL);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLTOALL);

    mpi_errno = MPIR_Alltoall_impl(sendbuf, sendcount, sendtype,
                                   recvbuf, recvcount, recvtype, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLTOALL);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_alltoallv(const void *sendbuf, const int *sendcounts,
                                                    const int *sdispls, MPI_Datatype sendtype,
                                                    void *recvbuf, const int *recvcounts,
                                                    const int *rdispls, MPI_Datatype recvtype,
                                                    MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLTOALLV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLTOALLV);

    mpi_errno = MPIR_Alltoallv_impl(sendbuf, sendcounts, sdispls,
                                    sendtype, recvbuf, recvcounts,
                                    rdispls, recvtype, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLTOALLV);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_alltoallw(const void *sendbuf, const int sendcounts[],
                                                    const int sdispls[],
                                                    const MPI_Datatype sendtypes[], void *recvbuf,
                                                    const int recvcounts[], const int rdispls[],
                                                    const MPI_Datatype recvtypes[],
                                                    MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ALLTOALLW);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ALLTOALLW);

    mpi_errno = MPIR_Alltoallw_impl(sendbuf, sendcounts, sdispls,
                                    sendtypes, recvbuf, recvcounts,
                                    rdispls, recvtypes, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ALLTOALLW);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_reduce(const void *sendbuf, void *recvbuf, int count,
                                                 MPI_Datatype datatype, MPI_Op op, int root,
                                                 MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_REDUCE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_REDUCE);

    mpi_errno = MPIR_Reduce_impl(sendbuf, recvbuf, count, datatype, op, root, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_REDUCE);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_reduce_scatter(const void *sendbuf, void *recvbuf,
                                                         const int recvcounts[],
                                                         MPI_Datatype datatype, MPI_Op op,
                                                         MPIR_Comm * comm, MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER);

    mpi_errno = MPIR_Reduce_scatter_impl(sendbuf, recvbuf, recvcounts, datatype, op, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_reduce_scatter_block(const void *sendbuf, void *recvbuf,
                                                               int recvcount, MPI_Datatype datatype,
                                                               MPI_Op op, MPIR_Comm * comm,
                                                               MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER_BLOCK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER_BLOCK);

    MPIR_Reduce_scatter_block_impl(sendbuf, recvbuf, recvcount, datatype, op, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_REDUCE_SCATTER_BLOCK);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_scan(const void *sendbuf, void *recvbuf, int count,
                                               MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                               MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_SCAN);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_SCAN);

    mpi_errno = MPIR_Scan_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_SCAN);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_exscan(const void *sendbuf, void *recvbuf, int count,
                                                 MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                                 MPIR_Errflag_t * errflag)
{
    int mpi_errno = MPI_SUCCESS;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_EXSCAN);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_EXSCAN);

    mpi_errno = MPIR_Exscan_impl(sendbuf, recvbuf, count, datatype, op, comm, errflag);

    MPIR_ERR_CHECK(mpi_errno);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_EXSCAN);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_neighbor_allgather(const void *sendbuf, int sendcount,
                                                             MPI_Datatype sendtype, void *recvbuf,
                                                             int recvcount, MPI_Datatype recvtype,
                                                             MPIR_Comm * comm)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHER);

    mpi_errno =
        MPIR_Neighbor_allgather_impl(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                     comm);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_neighbor_allgatherv(const void *sendbuf, int sendcount,
                                                              MPI_Datatype sendtype, void *recvbuf,
                                                              const int recvcounts[],
                                                              const int displs[],
                                                              MPI_Datatype recvtype,
                                                              MPIR_Comm * comm)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHERV);

    mpi_errno = MPIR_Neighbor_allgatherv_impl(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcounts, displs, recvtype, comm);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLGATHERV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_neighbor_alltoall(const void *sendbuf, int sendcount,
                                                            MPI_Datatype sendtype, void *recvbuf,
                                                            int recvcount, MPI_Datatype recvtype,
                                                            MPIR_Comm * comm)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALL);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALL);

    mpi_errno = MPIR_Neighbor_alltoall_impl(sendbuf, sendcount, sendtype,
                                            recvbuf, recvcount, recvtype, comm);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALL);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_neighbor_alltoallv(const void *sendbuf,
                                                             const int sendcounts[],
                                                             const int sdispls[],
                                                             MPI_Datatype sendtype, void *recvbuf,
                                                             const int recvcounts[],
                                                             const int rdispls[],
                                                             MPI_Datatype recvtype,
                                                             MPIR_Comm * comm)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLV);

    mpi_errno = MPIR_Neighbor_alltoallv_impl(sendbuf, sendcounts, sdispls, sendtype,
                                             recvbuf, recvcounts, rdispls, recvtype, comm);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_neighbor_alltoallw(const void *sendbuf,
                                                             const int sendcounts[],
                                                             const MPI_Aint sdispls[],
                                                             const MPI_Datatype sendtypes[],
                                                             void *recvbuf, const int recvcounts[],
                                                             const MPI_Aint rdispls[],
                                                             const MPI_Datatype recvtypes[],
                                                             MPIR_Comm * comm)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLW);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLW);

    mpi_errno = MPIR_Neighbor_alltoallw_impl(sendbuf, sendcounts, sdispls, sendtypes,
                                             recvbuf, recvcounts, rdispls, recvtypes, comm);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_NEIGHBOR_ALLTOALLW);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ineighbor_allgather(const void *sendbuf, int sendcount,
                                                              MPI_Datatype sendtype, void *recvbuf,
                                                              int recvcount, MPI_Datatype recvtype,
                                                              MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHER);

    mpi_errno = MPIR_Ineighbor_allgather_impl(sendbuf, sendcount, sendtype,
                                              recvbuf, recvcount, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ineighbor_allgatherv(const void *sendbuf, int sendcount,
                                                               MPI_Datatype sendtype, void *recvbuf,
                                                               const int recvcounts[],
                                                               const int displs[],
                                                               MPI_Datatype recvtype,
                                                               MPIR_Comm * comm,
                                                               MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHERV);

    mpi_errno = MPIR_Ineighbor_allgatherv_impl(sendbuf, sendcount, sendtype,
                                               recvbuf, recvcounts, displs, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLGATHERV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ineighbor_alltoall(const void *sendbuf, int sendcount,
                                                             MPI_Datatype sendtype, void *recvbuf,
                                                             int recvcount, MPI_Datatype recvtype,
                                                             MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALL);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALL);

    mpi_errno = MPIR_Ineighbor_alltoall_impl(sendbuf, sendcount, sendtype,
                                             recvbuf, recvcount, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALL);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ineighbor_alltoallv(const void *sendbuf,
                                                              const int sendcounts[],
                                                              const int sdispls[],
                                                              MPI_Datatype sendtype, void *recvbuf,
                                                              const int recvcounts[],
                                                              const int rdispls[],
                                                              MPI_Datatype recvtype,
                                                              MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLV);

    mpi_errno = MPIR_Ineighbor_alltoallv_impl(sendbuf, sendcounts, sdispls, sendtype,
                                              recvbuf, recvcounts, rdispls, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ineighbor_alltoallw(const void *sendbuf,
                                                              const int sendcounts[],
                                                              const MPI_Aint sdispls[],
                                                              const MPI_Datatype sendtypes[],
                                                              void *recvbuf, const int recvcounts[],
                                                              const MPI_Aint rdispls[],
                                                              const MPI_Datatype recvtypes[],
                                                              MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLW);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLW);

    mpi_errno = MPIR_Ineighbor_alltoallw_impl(sendbuf, sendcounts, sdispls, sendtypes,
                                              recvbuf, recvcounts, rdispls, recvtypes, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_INEIGHBOR_ALLTOALLW);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ibarrier(MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IBARRIER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IBARRIER);

    mpi_errno = MPIR_Ibarrier_impl(comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IBARRIER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ibcast(void *buffer, int count, MPI_Datatype datatype,
                                                 int root, MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IBCAST);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IBCAST);

    mpi_errno = MPIR_Ibcast_impl(buffer, count, datatype, root, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IBCAST);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iallgather(const void *sendbuf, int sendcount,
                                                     MPI_Datatype sendtype, void *recvbuf,
                                                     int recvcount, MPI_Datatype recvtype,
                                                     MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLGATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLGATHER);

    mpi_errno = MPIR_Iallgather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                     recvcount, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLGATHER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iallgatherv(const void *sendbuf, int sendcount,
                                                      MPI_Datatype sendtype, void *recvbuf,
                                                      const int *recvcounts, const int *displs,
                                                      MPI_Datatype recvtype, MPIR_Comm * comm,
                                                      MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLGATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLGATHERV);

    mpi_errno = MPIR_Iallgatherv_impl(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcounts, displs, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLGATHERV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iallreduce(const void *sendbuf, void *recvbuf, int count,
                                                     MPI_Datatype datatype, MPI_Op op,
                                                     MPIR_Comm * comm, MPIR_Request ** request)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLREDUCE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLREDUCE);

    mpi_errno = MPIR_Iallreduce_impl(sendbuf, recvbuf, count, datatype, op, comm, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLREDUCE);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ialltoall(const void *sendbuf, int sendcount,
                                                    MPI_Datatype sendtype, void *recvbuf,
                                                    int recvcount, MPI_Datatype recvtype,
                                                    MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLTOALL);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLTOALL);

    mpi_errno = MPIR_Ialltoall_impl(sendbuf, sendcount, sendtype, recvbuf,
                                    recvcount, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLTOALL);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ialltoallv(const void *sendbuf, const int *sendcounts,
                                                     const int *sdispls, MPI_Datatype sendtype,
                                                     void *recvbuf, const int *recvcounts,
                                                     const int *rdispls, MPI_Datatype recvtype,
                                                     MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLTOALLV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLTOALLV);

    mpi_errno = MPIR_Ialltoallv_impl(sendbuf, sendcounts, sdispls,
                                     sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLTOALLV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ialltoallw(const void *sendbuf, const int *sendcounts,
                                                     const int *sdispls,
                                                     const MPI_Datatype sendtypes[], void *recvbuf,
                                                     const int *recvcounts, const int *rdispls,
                                                     const MPI_Datatype recvtypes[],
                                                     MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IALLTOALLW);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IALLTOALLW);

    mpi_errno = MPIR_Ialltoallw_impl(sendbuf, sendcounts, sdispls,
                                     sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IALLTOALLW);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iexscan(const void *sendbuf, void *recvbuf, int count,
                                                  MPI_Datatype datatype, MPI_Op op,
                                                  MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IEXSCAN);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IEXSCAN);

    mpi_errno = MPIR_Iexscan_impl(sendbuf, recvbuf, count, datatype, op, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IEXSCAN);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_igather(const void *sendbuf, int sendcount,
                                                  MPI_Datatype sendtype, void *recvbuf,
                                                  int recvcount, MPI_Datatype recvtype, int root,
                                                  MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IGATHER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IGATHER);

    mpi_errno = MPIR_Igather_impl(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, root, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IGATHER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_igatherv(const void *sendbuf, int sendcount,
                                                   MPI_Datatype sendtype, void *recvbuf,
                                                   const int *recvcounts, const int *displs,
                                                   MPI_Datatype recvtype, int root,
                                                   MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IGATHERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IGATHERV);

    mpi_errno = MPIR_Igatherv_impl(sendbuf, sendcount, sendtype,
                                   recvbuf, recvcounts, displs, recvtype, root, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IGATHERV);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ireduce_scatter_block(const void *sendbuf, void *recvbuf,
                                                                int recvcount,
                                                                MPI_Datatype datatype, MPI_Op op,
                                                                MPIR_Comm * comm,
                                                                MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER_BLOCK);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER_BLOCK);

    mpi_errno = MPIR_Ireduce_scatter_block_impl(sendbuf, recvbuf, recvcount,
                                                datatype, op, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER_BLOCK);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ireduce_scatter(const void *sendbuf, void *recvbuf,
                                                          const int recvcounts[],
                                                          MPI_Datatype datatype, MPI_Op op,
                                                          MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER);

    mpi_errno = MPIR_Ireduce_scatter_impl(sendbuf, recvbuf, recvcounts, datatype, op, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IREDUCE_SCATTER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_ireduce(const void *sendbuf, void *recvbuf, int count,
                                                  MPI_Datatype datatype, MPI_Op op, int root,
                                                  MPIR_Comm * comm, MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_IREDUCE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_IREDUCE);

    mpi_errno = MPIR_Ireduce_impl(sendbuf, recvbuf, count, datatype, op, root, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_IREDUCE);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iscan(const void *sendbuf, void *recvbuf, int count,
                                                MPI_Datatype datatype, MPI_Op op, MPIR_Comm * comm,
                                                MPIR_Request ** req)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ISCAN);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ISCAN);

    mpi_errno = MPIR_Iscan_impl(sendbuf, recvbuf, count, datatype, op, comm, req);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ISCAN);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iscatter(const void *sendbuf, int sendcount,
                                                   MPI_Datatype sendtype, void *recvbuf,
                                                   int recvcount, MPI_Datatype recvtype,
                                                   int root, MPIR_Comm * comm,
                                                   MPIR_Request ** request)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ISCATTER);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ISCATTER);

    mpi_errno = MPIR_Iscatter_impl(sendbuf, sendcount, sendtype, recvbuf,
                                   recvcount, recvtype, root, comm, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ISCATTER);
    return mpi_errno;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_NM_mpi_iscatterv(const void *sendbuf, const int *sendcounts,
                                                    const int *displs, MPI_Datatype sendtype,
                                                    void *recvbuf, int recvcount,
                                                    MPI_Datatype recvtype, int root,
                                                    MPIR_Comm * comm, MPIR_Request ** request)
{
    int mpi_errno;
    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIDI_NM_MPI_ISCATTERV);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIDI_NM_MPI_ISCATTERV);

    mpi_errno = MPIR_Iscatterv_impl(sendbuf, sendcounts, displs, sendtype,
                                    recvbuf, recvcount, recvtype, root, comm, request);

    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIDI_NM_MPI_ISCATTERV);
    return mpi_errno;
}

#endif /* OFI_COLL_H_INCLUDED */