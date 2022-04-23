/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 *
 * This file is automatically generated by buildiface 
 * DO NOT EDIT
 */
#include "mpi_fortimpl.h"


/* Begin MPI profiling block */
#if defined(USE_WEAK_SYMBOLS) && !defined(USE_ONLY_MPI_NAMES) 
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_TYPE_CREATE_SUBARRAY = PMPI_TYPE_CREATE_SUBARRAY
#pragma weak mpi_type_create_subarray__ = PMPI_TYPE_CREATE_SUBARRAY
#pragma weak mpi_type_create_subarray_ = PMPI_TYPE_CREATE_SUBARRAY
#pragma weak mpi_type_create_subarray = PMPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray__
#pragma weak mpi_type_create_subarray__ = pmpi_type_create_subarray__
#pragma weak mpi_type_create_subarray_ = pmpi_type_create_subarray__
#pragma weak mpi_type_create_subarray = pmpi_type_create_subarray__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray_
#pragma weak mpi_type_create_subarray__ = pmpi_type_create_subarray_
#pragma weak mpi_type_create_subarray_ = pmpi_type_create_subarray_
#pragma weak mpi_type_create_subarray = pmpi_type_create_subarray_
#else
#pragma weak MPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray
#pragma weak mpi_type_create_subarray__ = pmpi_type_create_subarray
#pragma weak mpi_type_create_subarray_ = pmpi_type_create_subarray
#pragma weak mpi_type_create_subarray = pmpi_type_create_subarray
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_TYPE_CREATE_SUBARRAY = PMPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_type_create_subarray__ = pmpi_type_create_subarray__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_type_create_subarray = pmpi_type_create_subarray
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_type_create_subarray_ = pmpi_type_create_subarray_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_TYPE_CREATE_SUBARRAY  MPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_type_create_subarray__  mpi_type_create_subarray__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_type_create_subarray  mpi_type_create_subarray
#else
#pragma _HP_SECONDARY_DEF pmpi_type_create_subarray_  mpi_type_create_subarray_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_TYPE_CREATE_SUBARRAY as PMPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_type_create_subarray__ as pmpi_type_create_subarray__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_type_create_subarray as pmpi_type_create_subarray
#else
#pragma _CRI duplicate mpi_type_create_subarray_ as pmpi_type_create_subarray_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_type_create_subarray__ = MPI_TYPE_CREATE_SUBARRAY
#pragma weak mpi_type_create_subarray_ = MPI_TYPE_CREATE_SUBARRAY
#pragma weak mpi_type_create_subarray = MPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_TYPE_CREATE_SUBARRAY = mpi_type_create_subarray__
#pragma weak mpi_type_create_subarray_ = mpi_type_create_subarray__
#pragma weak mpi_type_create_subarray = mpi_type_create_subarray__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_TYPE_CREATE_SUBARRAY = mpi_type_create_subarray_
#pragma weak mpi_type_create_subarray__ = mpi_type_create_subarray_
#pragma weak mpi_type_create_subarray = mpi_type_create_subarray_
#else
#pragma weak MPI_TYPE_CREATE_SUBARRAY = mpi_type_create_subarray
#pragma weak mpi_type_create_subarray__ = mpi_type_create_subarray
#pragma weak mpi_type_create_subarray_ = mpi_type_create_subarray
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_TYPE_CREATE_SUBARRAY")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_type_create_subarray__ = PMPI_TYPE_CREATE_SUBARRAY
#pragma weak pmpi_type_create_subarray_ = PMPI_TYPE_CREATE_SUBARRAY
#pragma weak pmpi_type_create_subarray = PMPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray__
#pragma weak pmpi_type_create_subarray_ = pmpi_type_create_subarray__
#pragma weak pmpi_type_create_subarray = pmpi_type_create_subarray__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray_
#pragma weak pmpi_type_create_subarray__ = pmpi_type_create_subarray_
#pragma weak pmpi_type_create_subarray = pmpi_type_create_subarray_
#else
#pragma weak PMPI_TYPE_CREATE_SUBARRAY = pmpi_type_create_subarray
#pragma weak pmpi_type_create_subarray__ = pmpi_type_create_subarray
#pragma weak pmpi_type_create_subarray_ = pmpi_type_create_subarray
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_TYPE_CREATE_SUBARRAY")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_TYPE_CREATE_SUBARRAY( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray__( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_type_create_subarray_( MPI_Fint *, MPI_Fint [], MPI_Fint [], MPI_Fint [], MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_type_create_subarray")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_type_create_subarray_ PMPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_type_create_subarray_ pmpi_type_create_subarray__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_type_create_subarray_ pmpi_type_create_subarray
#else
#define mpi_type_create_subarray_ pmpi_type_create_subarray_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Type_create_subarray
#define MPI_Type_create_subarray PMPI_Type_create_subarray 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_type_create_subarray_ MPI_TYPE_CREATE_SUBARRAY
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_type_create_subarray_ mpi_type_create_subarray__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_type_create_subarray_ mpi_type_create_subarray
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_type_create_subarray_ ( MPI_Fint *v1, MPI_Fint v2[], MPI_Fint v3[], MPI_Fint v4[], MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *v7, MPI_Fint *ierr ){
    *ierr = MPI_Type_create_subarray( (int)*v1, v2, v3, v4, (int)*v5, (MPI_Datatype)(*v6), (MPI_Datatype *)(v7) );
}
