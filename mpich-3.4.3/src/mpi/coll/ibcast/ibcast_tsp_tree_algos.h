/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* Header protection (i.e., IBCAST_TSP_TREE_ALGOS_H_INCLUDED) is
 * intentionally omitted since this header might get included multiple
 * times within the same .c file. */

#include "algo_common.h"
#include "treealgo.h"
#include "tsp_namespace_def.h"

/* Routine to schedule a pipelined tree based broadcast */
int MPIR_TSP_Ibcast_sched_intra_tree(void *buffer, int count, MPI_Datatype datatype, int root,
                                     MPIR_Comm * comm, int tree_type, int k, int chunk_size,
                                     MPIR_TSP_sched_t * sched)
{
    int mpi_errno = MPI_SUCCESS;
    int i;
    MPI_Aint num_chunks, chunk_size_floor, chunk_size_ceil;
    int offset = 0;
    size_t extent, type_size;
    MPI_Aint lb, true_extent;
    int size;
    int rank;
    int recv_id;
    int num_children;
    MPIR_Treealgo_tree_t my_tree;
    int tag;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IBCAST_SCHED_INTRA_TREE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IBCAST_SCHED_INTRA_TREE);

    MPL_DBG_MSG_FMT(MPIR_DBG_COLL, VERBOSE,
                    (MPL_DBG_FDEST, "Scheduling pipelined tree broadcast on %d ranks, root=%d",
                     MPIR_Comm_size(comm), root));

    size = MPIR_Comm_size(comm);
    rank = MPIR_Comm_rank(comm);

    MPIR_Datatype_get_size_macro(datatype, type_size);
    MPIR_Datatype_get_extent_macro(datatype, extent);
    MPIR_Type_get_true_extent_impl(datatype, &lb, &true_extent);
    extent = MPL_MAX(extent, true_extent);

    /* calculate chunking information for pipelining */
    MPIR_Algo_calculate_pipeline_chunk_info(chunk_size, type_size, count, &num_chunks,
                                            &chunk_size_floor, &chunk_size_ceil);
    /* print chunking information */
    MPL_DBG_MSG_FMT(MPIR_DBG_COLL, VERBOSE, (MPL_DBG_FDEST,
                                             "Broadcast pipeline info: chunk_size=%d count=%d num_chunks=%d chunk_size_floor=%d chunk_size_ceil=%d",
                                             chunk_size, count, num_chunks,
                                             chunk_size_floor, chunk_size_ceil));

    mpi_errno = MPIR_Treealgo_tree_create(rank, size, tree_type, k, root, &my_tree);
    MPIR_ERR_CHECK(mpi_errno);
    num_children = my_tree.num_children;

    /* do pipelined tree broadcast */
    /* NOTE: Make sure you are handling non-contiguous datatypes
     * correctly with pipelined broadcast, for example, buffer+offset
     * if being calculated correctly */
    for (i = 0; i < num_chunks; i++) {
        int msgsize = (i == 0) ? chunk_size_floor : chunk_size_ceil;

        /* For correctness, transport based collectives need to get the
         * tag from the same pool as schedule based collectives */
        mpi_errno = MPIR_Sched_next_tag(comm, &tag);
        MPIR_ERR_CHECK(mpi_errno);

        /* Receive message from parent */
        if (my_tree.parent != -1) {
            recv_id =
                MPIR_TSP_sched_irecv((char *) buffer + offset * extent, msgsize, datatype,
                                     my_tree.parent, tag, comm, sched, 0, NULL);
        }

        if (num_children) {
            /* Multicast data to the children */
            MPIR_TSP_sched_imcast((char *) buffer + offset * extent, msgsize, datatype,
                                  my_tree.children, num_children, tag, comm, sched,
                                  (my_tree.parent != -1) ? 1 : 0, &recv_id);
        }
        offset += msgsize;
    }

    MPIR_Treealgo_tree_free(&my_tree);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIR_TSP_IBCAST_SCHED_INTRA_TREE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}


/* Non-blocking tree based broadcast */
int MPIR_TSP_Ibcast_intra_tree(void *buffer, int count, MPI_Datatype datatype, int root,
                               MPIR_Comm * comm, MPIR_Request ** req, int tree_type, int k,
                               int chunk_size)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_TSP_sched_t *sched;

    MPIR_FUNC_VERBOSE_STATE_DECL(MPID_STATE_MPIR_TSP_IBCAST_INTRA_TREE);
    MPIR_FUNC_VERBOSE_ENTER(MPID_STATE_MPIR_TSP_IBCAST_INTRA_TREE);

    *req = NULL;

    /* generate the schedule */
    sched = MPL_malloc(sizeof(MPIR_TSP_sched_t), MPL_MEM_COLL);
    MPIR_Assert(sched != NULL);
    MPIR_TSP_sched_create(sched);

    /* schedule pipelined tree algo */
    mpi_errno = MPIR_TSP_Ibcast_sched_intra_tree(buffer, count, datatype, root, comm,
                                                 tree_type, k, chunk_size, sched);
    MPIR_ERR_CHECK(mpi_errno);

    /* start and register the schedule */
    mpi_errno = MPIR_TSP_sched_start(sched, comm, req);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_FUNC_VERBOSE_EXIT(MPID_STATE_MPIR_TSP_IBCAST_INTRA_TREE);
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
