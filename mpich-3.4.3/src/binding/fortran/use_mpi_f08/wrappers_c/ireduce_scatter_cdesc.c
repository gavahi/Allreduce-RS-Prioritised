/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Ireduce_scatter_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, const int x2[], MPI_Datatype x3, MPI_Op x4, MPI_Comm x5, MPI_Request * x6)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    void *buf1 = x1->base_addr;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    } else if (buf0 == &MPIR_F08_MPI_IN_PLACE) {
        buf0 = MPI_IN_PLACE;
    }

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    err = MPI_Ireduce_scatter(buf0, buf1, x2, x3, x4, x5, x6);

    return err;
}
