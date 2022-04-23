/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Ialltoallv_cdesc(CFI_cdesc_t* x0, const int x1[], const int x2[], MPI_Datatype x3, CFI_cdesc_t* x4, const int x5[], const int x6[], MPI_Datatype x7, MPI_Comm x8, MPI_Request * x9)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    void *buf4 = x4->base_addr;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    } else if (buf0 == &MPIR_F08_MPI_IN_PLACE) {
        buf0 = MPI_IN_PLACE;
    }

    if (buf4 == &MPIR_F08_MPI_BOTTOM) {
        buf4 = MPI_BOTTOM;
    }

    err = MPI_Ialltoallv(buf0, x1, x2, x3, buf4, x5, x6, x7, x8, x9);

    return err;
}
