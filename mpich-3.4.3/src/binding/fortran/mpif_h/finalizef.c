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
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_FINALIZE = PMPI_FINALIZE
#pragma weak mpi_finalize__ = PMPI_FINALIZE
#pragma weak mpi_finalize_ = PMPI_FINALIZE
#pragma weak mpi_finalize = PMPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FINALIZE = pmpi_finalize__
#pragma weak mpi_finalize__ = pmpi_finalize__
#pragma weak mpi_finalize_ = pmpi_finalize__
#pragma weak mpi_finalize = pmpi_finalize__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FINALIZE = pmpi_finalize_
#pragma weak mpi_finalize__ = pmpi_finalize_
#pragma weak mpi_finalize_ = pmpi_finalize_
#pragma weak mpi_finalize = pmpi_finalize_
#else
#pragma weak MPI_FINALIZE = pmpi_finalize
#pragma weak mpi_finalize__ = pmpi_finalize
#pragma weak mpi_finalize_ = pmpi_finalize
#pragma weak mpi_finalize = pmpi_finalize
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * );

#pragma weak MPI_FINALIZE = PMPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * );

#pragma weak mpi_finalize__ = pmpi_finalize__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * );

#pragma weak mpi_finalize = pmpi_finalize
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * );

#pragma weak mpi_finalize_ = pmpi_finalize_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_FINALIZE  MPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_finalize__  mpi_finalize__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_finalize  mpi_finalize
#else
#pragma _HP_SECONDARY_DEF pmpi_finalize_  mpi_finalize_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_FINALIZE as PMPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_finalize__ as pmpi_finalize__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_finalize as pmpi_finalize
#else
#pragma _CRI duplicate mpi_finalize_ as pmpi_finalize_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_finalize__ = MPI_FINALIZE
#pragma weak mpi_finalize_ = MPI_FINALIZE
#pragma weak mpi_finalize = MPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FINALIZE = mpi_finalize__
#pragma weak mpi_finalize_ = mpi_finalize__
#pragma weak mpi_finalize = mpi_finalize__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FINALIZE = mpi_finalize_
#pragma weak mpi_finalize__ = mpi_finalize_
#pragma weak mpi_finalize = mpi_finalize_
#else
#pragma weak MPI_FINALIZE = mpi_finalize
#pragma weak mpi_finalize__ = mpi_finalize
#pragma weak mpi_finalize_ = mpi_finalize
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("MPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("MPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("MPI_FINALIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("mpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL mpi_finalize( MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_FINALIZE( MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize__( MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize_( MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize( MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_finalize__ = PMPI_FINALIZE
#pragma weak pmpi_finalize_ = PMPI_FINALIZE
#pragma weak pmpi_finalize = PMPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_FINALIZE = pmpi_finalize__
#pragma weak pmpi_finalize_ = pmpi_finalize__
#pragma weak pmpi_finalize = pmpi_finalize__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_FINALIZE = pmpi_finalize_
#pragma weak pmpi_finalize__ = pmpi_finalize_
#pragma weak pmpi_finalize = pmpi_finalize_
#else
#pragma weak PMPI_FINALIZE = pmpi_finalize
#pragma weak pmpi_finalize__ = pmpi_finalize
#pragma weak pmpi_finalize_ = pmpi_finalize
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize( MPI_Fint * ) __attribute__((weak,alias("PMPI_FINALIZE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_FINALIZE( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize__( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_finalize_( MPI_Fint * ) __attribute__((weak,alias("pmpi_finalize")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_finalize_ PMPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_finalize_ pmpi_finalize__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_finalize_ pmpi_finalize
#else
#define mpi_finalize_ pmpi_finalize_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Finalize
#define MPI_Finalize PMPI_Finalize 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_finalize_ MPI_FINALIZE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_finalize_ mpi_finalize__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_finalize_ mpi_finalize
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_finalize_ ( MPI_Fint *ierr ){
    *ierr = MPI_Finalize(  );
}
