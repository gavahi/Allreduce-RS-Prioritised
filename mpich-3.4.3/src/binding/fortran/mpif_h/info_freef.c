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
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_INFO_FREE = PMPI_INFO_FREE
#pragma weak mpi_info_free__ = PMPI_INFO_FREE
#pragma weak mpi_info_free_ = PMPI_INFO_FREE
#pragma weak mpi_info_free = PMPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INFO_FREE = pmpi_info_free__
#pragma weak mpi_info_free__ = pmpi_info_free__
#pragma weak mpi_info_free_ = pmpi_info_free__
#pragma weak mpi_info_free = pmpi_info_free__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INFO_FREE = pmpi_info_free_
#pragma weak mpi_info_free__ = pmpi_info_free_
#pragma weak mpi_info_free_ = pmpi_info_free_
#pragma weak mpi_info_free = pmpi_info_free_
#else
#pragma weak MPI_INFO_FREE = pmpi_info_free
#pragma weak mpi_info_free__ = pmpi_info_free
#pragma weak mpi_info_free_ = pmpi_info_free
#pragma weak mpi_info_free = pmpi_info_free
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * );

#pragma weak MPI_INFO_FREE = PMPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * );

#pragma weak mpi_info_free__ = pmpi_info_free__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * );

#pragma weak mpi_info_free = pmpi_info_free
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * );

#pragma weak mpi_info_free_ = pmpi_info_free_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_INFO_FREE  MPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_info_free__  mpi_info_free__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_info_free  mpi_info_free
#else
#pragma _HP_SECONDARY_DEF pmpi_info_free_  mpi_info_free_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_INFO_FREE as PMPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_info_free__ as pmpi_info_free__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_info_free as pmpi_info_free
#else
#pragma _CRI duplicate mpi_info_free_ as pmpi_info_free_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_info_free__ = MPI_INFO_FREE
#pragma weak mpi_info_free_ = MPI_INFO_FREE
#pragma weak mpi_info_free = MPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INFO_FREE = mpi_info_free__
#pragma weak mpi_info_free_ = mpi_info_free__
#pragma weak mpi_info_free = mpi_info_free__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INFO_FREE = mpi_info_free_
#pragma weak mpi_info_free__ = mpi_info_free_
#pragma weak mpi_info_free = mpi_info_free_
#else
#pragma weak MPI_INFO_FREE = mpi_info_free
#pragma weak mpi_info_free__ = mpi_info_free
#pragma weak mpi_info_free_ = mpi_info_free
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INFO_FREE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_free( MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_FREE( MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free__( MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free_( MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free( MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_info_free__ = PMPI_INFO_FREE
#pragma weak pmpi_info_free_ = PMPI_INFO_FREE
#pragma weak pmpi_info_free = PMPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_INFO_FREE = pmpi_info_free__
#pragma weak pmpi_info_free_ = pmpi_info_free__
#pragma weak pmpi_info_free = pmpi_info_free__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_INFO_FREE = pmpi_info_free_
#pragma weak pmpi_info_free__ = pmpi_info_free_
#pragma weak pmpi_info_free = pmpi_info_free_
#else
#pragma weak PMPI_INFO_FREE = pmpi_info_free
#pragma weak pmpi_info_free__ = pmpi_info_free
#pragma weak pmpi_info_free_ = pmpi_info_free
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INFO_FREE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_FREE( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free__( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_free_( MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_info_free")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_info_free_ PMPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_info_free_ pmpi_info_free__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_info_free_ pmpi_info_free
#else
#define mpi_info_free_ pmpi_info_free_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Info_free
#define MPI_Info_free PMPI_Info_free 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_info_free_ MPI_INFO_FREE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_info_free_ mpi_info_free__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_info_free_ mpi_info_free
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_info_free_ ( MPI_Fint *v1, MPI_Fint *ierr ){
    *ierr = MPI_Info_free( (MPI_Info *)(v1) );
}
