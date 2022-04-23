/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Exscan_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPI_Op x4, MPI_Comm x5)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    void *buf1 = x1->base_addr;
    int count1 = x2;
    MPI_Datatype dtype1 = x3;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    } else if (buf0 == &MPIR_F08_MPI_IN_PLACE) {
        buf0 = MPI_IN_PLACE;
    }

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    if (x1->rank != 0 && !CFI_is_contiguous(x1)) {
        err = cdesc_create_datatype(x1, x2, x3, &dtype1);
        count1 = 1;
    }

    err = MPI_Exscan(buf0, buf1, count1, dtype1, x4, x5);

    if (dtype1 != x3)  MPI_Type_free(&dtype1);
    return err;
}
