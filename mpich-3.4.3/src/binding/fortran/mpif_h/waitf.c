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
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_WAIT = PMPI_WAIT
#pragma weak mpi_wait__ = PMPI_WAIT
#pragma weak mpi_wait_ = PMPI_WAIT
#pragma weak mpi_wait = PMPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WAIT = pmpi_wait__
#pragma weak mpi_wait__ = pmpi_wait__
#pragma weak mpi_wait_ = pmpi_wait__
#pragma weak mpi_wait = pmpi_wait__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WAIT = pmpi_wait_
#pragma weak mpi_wait__ = pmpi_wait_
#pragma weak mpi_wait_ = pmpi_wait_
#pragma weak mpi_wait = pmpi_wait_
#else
#pragma weak MPI_WAIT = pmpi_wait
#pragma weak mpi_wait__ = pmpi_wait
#pragma weak mpi_wait_ = pmpi_wait
#pragma weak mpi_wait = pmpi_wait
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_WAIT = PMPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_wait__ = pmpi_wait__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_wait = pmpi_wait
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_wait_ = pmpi_wait_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_WAIT  MPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_wait__  mpi_wait__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_wait  mpi_wait
#else
#pragma _HP_SECONDARY_DEF pmpi_wait_  mpi_wait_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_WAIT as PMPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_wait__ as pmpi_wait__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_wait as pmpi_wait
#else
#pragma _CRI duplicate mpi_wait_ as pmpi_wait_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_wait__ = MPI_WAIT
#pragma weak mpi_wait_ = MPI_WAIT
#pragma weak mpi_wait = MPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_WAIT = mpi_wait__
#pragma weak mpi_wait_ = mpi_wait__
#pragma weak mpi_wait = mpi_wait__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_WAIT = mpi_wait_
#pragma weak mpi_wait__ = mpi_wait_
#pragma weak mpi_wait = mpi_wait_
#else
#pragma weak MPI_WAIT = mpi_wait
#pragma weak mpi_wait__ = mpi_wait
#pragma weak mpi_wait_ = mpi_wait
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_WAIT")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL mpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_wait__ = PMPI_WAIT
#pragma weak pmpi_wait_ = PMPI_WAIT
#pragma weak pmpi_wait = PMPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_WAIT = pmpi_wait__
#pragma weak pmpi_wait_ = pmpi_wait__
#pragma weak pmpi_wait = pmpi_wait__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_WAIT = pmpi_wait_
#pragma weak pmpi_wait__ = pmpi_wait_
#pragma weak pmpi_wait = pmpi_wait_
#else
#pragma weak PMPI_WAIT = pmpi_wait
#pragma weak pmpi_wait__ = pmpi_wait
#pragma weak pmpi_wait_ = pmpi_wait
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_WAIT")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_WAIT( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait__( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_wait_( MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_wait")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_wait_ PMPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_wait_ pmpi_wait__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_wait_ pmpi_wait
#else
#define mpi_wait_ pmpi_wait_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Wait
#define MPI_Wait PMPI_Wait 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_wait_ MPI_WAIT
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_wait_ mpi_wait__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_wait_ mpi_wait
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_wait_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint *ierr ){

#ifndef HAVE_MPI_F_INIT_WORKS_WITH_C
    if (MPIR_F_NeedInit){ mpirinitf_(); MPIR_F_NeedInit = 0; }
#endif

    if (v2 == MPI_F_STATUS_IGNORE) { v2 = (MPI_Fint*)MPI_STATUS_IGNORE; }
    *ierr = MPI_Wait( (MPI_Request *)(v1), (MPI_Status *)v2 );
}
