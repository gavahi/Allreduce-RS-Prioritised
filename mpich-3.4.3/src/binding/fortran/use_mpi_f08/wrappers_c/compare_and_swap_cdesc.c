/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_Compare_and_swap_cdesc(CFI_cdesc_t* x0, CFI_cdesc_t* x1, CFI_cdesc_t* x2, MPI_Datatype x3, int x4, MPI_Aint x5, MPI_Win x6)
{
    int err = MPI_SUCCESS;
    void *buf0 = x0->base_addr;
    void *buf1 = x1->base_addr;
    void *buf2 = x2->base_addr;

    if (buf0 == &MPIR_F08_MPI_BOTTOM) {
        buf0 = MPI_BOTTOM;
    }

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    if (buf2 == &MPIR_F08_MPI_BOTTOM) {
        buf2 = MPI_BOTTOM;
    }

    err = MPI_Compare_and_swap(buf0, buf1, buf2, x3, x4, x5, x6);

    return err;
}
