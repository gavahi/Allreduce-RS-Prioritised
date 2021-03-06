/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Fetch_and_op_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, MPI_Datatype x2, int x3, MPI_Aint x4, MPI_Op x5, MPI_Win x6)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    void *buf1 = x1->base_addr;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    }

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    err = MPI_Fetch_and_op(buf0, buf1, x2, x3, x4, x5, x6);

    return err;
}
