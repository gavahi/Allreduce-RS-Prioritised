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
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_WIN_ALLOCATE = PMPI_WIN_ALLOCATE
#pragma weak mpi_win_allocate__ = PMPI_WIN_ALLOCATE
#pragma weak mpi_win_allocate_ = PMPI_WIN_ALLOCATE
#pragma weak mpi_win_allocate = PMPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WIN_ALLOCATE = pmpi_win_allocate__
#pragma weak mpi_win_allocate__ = pmpi_win_allocate__
#pragma weak mpi_win_allocate_ = pmpi_win_allocate__
#pragma weak mpi_win_allocate = pmpi_win_allocate__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WIN_ALLOCATE = pmpi_win_allocate_
#pragma weak mpi_win_allocate__ = pmpi_win_allocate_
#pragma weak mpi_win_allocate_ = pmpi_win_allocate_
#pragma weak mpi_win_allocate = pmpi_win_allocate_
#else
#pragma weak MPI_WIN_ALLOCATE = pmpi_win_allocate
#pragma weak mpi_win_allocate__ = pmpi_win_allocate
#pragma weak mpi_win_allocate_ = pmpi_win_allocate
#pragma weak mpi_win_allocate = pmpi_win_allocate
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_WIN_ALLOCATE = PMPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_allocate__ = pmpi_win_allocate__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_allocate = pmpi_win_allocate
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_win_allocate_ = pmpi_win_allocate_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_WIN_ALLOCATE  MPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_win_allocate__  mpi_win_allocate__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_win_allocate  mpi_win_allocate
#else
#pragma _HP_SECONDARY_DEF pmpi_win_allocate_  mpi_win_allocate_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_WIN_ALLOCATE as PMPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_win_allocate__ as pmpi_win_allocate__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_win_allocate as pmpi_win_allocate
#else
#pragma _CRI duplicate mpi_win_allocate_ as pmpi_win_allocate_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_win_allocate__ = MPI_WIN_ALLOCATE
#pragma weak mpi_win_allocate_ = MPI_WIN_ALLOCATE
#pragma weak mpi_win_allocate = MPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WIN_ALLOCATE = mpi_win_allocate__
#pragma weak mpi_win_allocate_ = mpi_win_allocate__
#pragma weak mpi_win_allocate = mpi_win_allocate__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WIN_ALLOCATE = mpi_win_allocate_
#pragma weak mpi_win_allocate__ = mpi_win_allocate_
#pragma weak mpi_win_allocate = mpi_win_allocate_
#else
#pragma weak MPI_WIN_ALLOCATE = mpi_win_allocate
#pragma weak mpi_win_allocate__ = mpi_win_allocate
#pragma weak mpi_win_allocate_ = mpi_win_allocate
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WIN_ALLOCATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL mpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_win_allocate__ = PMPI_WIN_ALLOCATE
#pragma weak pmpi_win_allocate_ = PMPI_WIN_ALLOCATE
#pragma weak pmpi_win_allocate = PMPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_WIN_ALLOCATE = pmpi_win_allocate__
#pragma weak pmpi_win_allocate_ = pmpi_win_allocate__
#pragma weak pmpi_win_allocate = pmpi_win_allocate__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_WIN_ALLOCATE = pmpi_win_allocate_
#pragma weak pmpi_win_allocate__ = pmpi_win_allocate_
#pragma weak pmpi_win_allocate = pmpi_win_allocate_
#else
#pragma weak PMPI_WIN_ALLOCATE = pmpi_win_allocate
#pragma weak pmpi_win_allocate__ = pmpi_win_allocate
#pragma weak pmpi_win_allocate_ = pmpi_win_allocate
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WIN_ALLOCATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_WIN_ALLOCATE( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate__( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_win_allocate_( MPI_Aint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, void*, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_win_allocate")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_win_allocate_ PMPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_win_allocate_ pmpi_win_allocate__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_win_allocate_ pmpi_win_allocate
#else
#define mpi_win_allocate_ pmpi_win_allocate_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Win_allocate
#define MPI_Win_allocate PMPI_Win_allocate 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_win_allocate_ MPI_WIN_ALLOCATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_win_allocate_ mpi_win_allocate__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_win_allocate_ mpi_win_allocate
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_win_allocate_ ( MPI_Aint * v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, void*v5, MPI_Fint *v6, MPI_Fint *ierr ){
    if (v5 == MPIR_F_MPI_BOTTOM) v5 = MPI_BOTTOM;
    *ierr = MPI_Win_allocate( *v1, (int)*v2, (MPI_Info)(*v3), (MPI_Comm)(*v4), v5, v6 );
}
