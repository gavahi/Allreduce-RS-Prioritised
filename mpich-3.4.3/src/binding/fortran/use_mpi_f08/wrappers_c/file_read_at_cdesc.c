/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_File_read_at_cdesc(MPI_File x0, MPI_Offset x1, CFI_cdesc_t* x2, int x3, MPI_Datatype x4, MPI_Status * x5)
{
    int err = MPI_SUCCESS;
#ifdef MPI_MODE_RDONLY
    void *buf2 = x2->base_addr;
    int count2 = x3;
    MPI_Datatype dtype2 = x4;

    if (buf2 == &MPIR_F08_MPI_BOTTOM) {
        buf2 = MPI_BOTTOM;
    }

    if (x2->rank != 0 && !CFI_is_contiguous(x2)) {
        err = cdesc_create_datatype(x2, x3, x4, &dtype2);
        count2 = 1;
    }

    err = MPI_File_read_at(x0, x1, buf2, count2, dtype2, x5);

    if (dtype2 != x4)  MPI_Type_free(&dtype2);
#else
    err = MPI_ERR_INTERN;
#endif
    return err;
}