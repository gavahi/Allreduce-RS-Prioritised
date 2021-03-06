/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Rsend_init_cdesc(CFI_cdesc_t* x0, int x1, MPI_Datatype x2, int x3, int x4, MPI_Comm x5, MPI_Request * x6)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    int count0 = x1;
    MPI_Datatype dtype0 = x2;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    }

    if (x0->rank != 0 && !CFI_is_contiguous(x0)) {
        err = cdesc_create_datatype(x0, x1, x2, &dtype0);
        count0 = 1;
    }

    err = MPI_Rsend_init(buf0, count0, dtype0, x3, x4, x5, x6);

    if (dtype0 != x2)  MPI_Type_free(&dtype0);
    return err;
}
