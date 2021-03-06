/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_File_iwrite_shared_cdesc(MPI_File x0, CFI_cdesc_t* x1, int x2, MPI_Datatype x3, MPIO_Request * x4)
{
    int err = MPI_SUCCESS;
#ifdef MPI_MODE_RDONLY
    void *buf1 = x1->base_addr;
    int count1 = x2;
    MPI_Datatype dtype1 = x3;

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    if (x1->rank != 0 && !CFI_is_contiguous(x1)) {
        err = cdesc_create_datatype(x1, x2, x3, &dtype1);
        count1 = 1;
    }

    err = MPI_File_iwrite_shared(x0, buf1, count1, dtype1, x4);

    if (dtype1 != x3)  MPI_Type_free(&dtype1);
#else
    err = MPI_ERR_INTERN;
#endif
    return err;
}
