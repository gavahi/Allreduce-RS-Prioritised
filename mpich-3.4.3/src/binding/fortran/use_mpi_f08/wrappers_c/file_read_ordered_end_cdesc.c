/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface
 * DO NOT EDIT
 */

#include "cdesc.h"

int MPIR_File_read_ordered_end_cdesc(MPI_File x0, CFI_cdesc_t* x1, MPI_Status * x2)
{
    int err = MPI_SUCCESS;
#ifdef MPI_MODE_RDONLY
    void *buf1 = x1->base_addr;

    if (buf1 == &MPIR_F08_MPI_BOTTOM) {
        buf1 = MPI_BOTTOM;
    }

    err = MPI_File_read_ordered_end(x0, buf1, x2);

#else
    err = MPI_ERR_INTERN;
#endif
    return err;
}
