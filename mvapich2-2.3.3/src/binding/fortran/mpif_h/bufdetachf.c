/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*  
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 *
 * This file is automatically generated by buildiface 
 * DO NOT EDIT
 */
#include "mpi_fortimpl.h"


/* Begin MPI profiling block */
#if defined(USE_WEAK_SYMBOLS) && !defined(USE_ONLY_MPI_NAMES) 
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_BUFFER_DETACH = PMPI_BUFFER_DETACH
#pragma weak mpi_buffer_detach__ = PMPI_BUFFER_DETACH
#pragma weak mpi_buffer_detach_ = PMPI_BUFFER_DETACH
#pragma weak mpi_buffer_detach = PMPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_BUFFER_DETACH = pmpi_buffer_detach__
#pragma weak mpi_buffer_detach__ = pmpi_buffer_detach__
#pragma weak mpi_buffer_detach_ = pmpi_buffer_detach__
#pragma weak mpi_buffer_detach = pmpi_buffer_detach__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_BUFFER_DETACH = pmpi_buffer_detach_
#pragma weak mpi_buffer_detach__ = pmpi_buffer_detach_
#pragma weak mpi_buffer_detach_ = pmpi_buffer_detach_
#pragma weak mpi_buffer_detach = pmpi_buffer_detach_
#else
#pragma weak MPI_BUFFER_DETACH = pmpi_buffer_detach
#pragma weak mpi_buffer_detach__ = pmpi_buffer_detach
#pragma weak mpi_buffer_detach_ = pmpi_buffer_detach
#pragma weak mpi_buffer_detach = pmpi_buffer_detach
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_BUFFER_DETACH = PMPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_buffer_detach__ = pmpi_buffer_detach__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_buffer_detach = pmpi_buffer_detach
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_buffer_detach_ = pmpi_buffer_detach_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_BUFFER_DETACH  MPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_buffer_detach__  mpi_buffer_detach__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_buffer_detach  mpi_buffer_detach
#else
#pragma _HP_SECONDARY_DEF pmpi_buffer_detach_  mpi_buffer_detach_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_BUFFER_DETACH as PMPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_buffer_detach__ as pmpi_buffer_detach__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_buffer_detach as pmpi_buffer_detach
#else
#pragma _CRI duplicate mpi_buffer_detach_ as pmpi_buffer_detach_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_buffer_detach__ = MPI_BUFFER_DETACH
#pragma weak mpi_buffer_detach_ = MPI_BUFFER_DETACH
#pragma weak mpi_buffer_detach = MPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_BUFFER_DETACH = mpi_buffer_detach__
#pragma weak mpi_buffer_detach_ = mpi_buffer_detach__
#pragma weak mpi_buffer_detach = mpi_buffer_detach__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_BUFFER_DETACH = mpi_buffer_detach_
#pragma weak mpi_buffer_detach__ = mpi_buffer_detach_
#pragma weak mpi_buffer_detach = mpi_buffer_detach_
#else
#pragma weak MPI_BUFFER_DETACH = mpi_buffer_detach
#pragma weak mpi_buffer_detach__ = mpi_buffer_detach
#pragma weak mpi_buffer_detach_ = mpi_buffer_detach
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_BUFFER_DETACH")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_buffer_detach__ = PMPI_BUFFER_DETACH
#pragma weak pmpi_buffer_detach_ = PMPI_BUFFER_DETACH
#pragma weak pmpi_buffer_detach = PMPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_BUFFER_DETACH = pmpi_buffer_detach__
#pragma weak pmpi_buffer_detach_ = pmpi_buffer_detach__
#pragma weak pmpi_buffer_detach = pmpi_buffer_detach__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_BUFFER_DETACH = pmpi_buffer_detach_
#pragma weak pmpi_buffer_detach__ = pmpi_buffer_detach_
#pragma weak pmpi_buffer_detach = pmpi_buffer_detach_
#else
#pragma weak PMPI_BUFFER_DETACH = pmpi_buffer_detach
#pragma weak pmpi_buffer_detach__ = pmpi_buffer_detach
#pragma weak pmpi_buffer_detach_ = pmpi_buffer_detach
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_BUFFER_DETACH")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_BUFFER_DETACH( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach__( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_buffer_detach_( void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_buffer_detach")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_buffer_detach_ PMPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_buffer_detach_ pmpi_buffer_detach__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_buffer_detach_ pmpi_buffer_detach
#else
#define mpi_buffer_detach_ pmpi_buffer_detach_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Buffer_detach
#define MPI_Buffer_detach PMPI_Buffer_detach 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_buffer_detach_ MPI_BUFFER_DETACH
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_buffer_detach_ mpi_buffer_detach__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_buffer_detach_ mpi_buffer_detach
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_buffer_detach_ ( void*v1, MPI_Fint *v2, MPI_Fint *ierr ){
    void *t1 = v1;
    if (v1 == MPIR_F_MPI_BOTTOM) v1 = MPI_BOTTOM;
    *ierr = MPI_Buffer_detach( &t1, v2 );
}
