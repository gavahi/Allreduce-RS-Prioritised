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
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *);

#if defined(F77_NAME_UPPER)
#pragma weak MPI_AINT_DIFF = PMPI_AINT_DIFF
#pragma weak mpi_aint_diff__ = PMPI_AINT_DIFF
#pragma weak mpi_aint_diff_ = PMPI_AINT_DIFF
#pragma weak mpi_aint_diff = PMPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_AINT_DIFF = pmpi_aint_diff__
#pragma weak mpi_aint_diff__ = pmpi_aint_diff__
#pragma weak mpi_aint_diff_ = pmpi_aint_diff__
#pragma weak mpi_aint_diff = pmpi_aint_diff__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_AINT_DIFF = pmpi_aint_diff_
#pragma weak mpi_aint_diff__ = pmpi_aint_diff_
#pragma weak mpi_aint_diff_ = pmpi_aint_diff_
#pragma weak mpi_aint_diff = pmpi_aint_diff_
#else
#pragma weak MPI_AINT_DIFF = pmpi_aint_diff
#pragma weak mpi_aint_diff__ = pmpi_aint_diff
#pragma weak mpi_aint_diff_ = pmpi_aint_diff
#pragma weak mpi_aint_diff = pmpi_aint_diff
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *);

#pragma weak MPI_AINT_DIFF = PMPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *);

#pragma weak mpi_aint_diff__ = pmpi_aint_diff__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *);

#pragma weak mpi_aint_diff = pmpi_aint_diff
#else
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *);

#pragma weak mpi_aint_diff_ = pmpi_aint_diff_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_AINT_DIFF  MPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_aint_diff__  mpi_aint_diff__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_aint_diff  mpi_aint_diff
#else
#pragma _HP_SECONDARY_DEF pmpi_aint_diff_  mpi_aint_diff_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_AINT_DIFF as PMPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_aint_diff__ as pmpi_aint_diff__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_aint_diff as pmpi_aint_diff
#else
#pragma _CRI duplicate mpi_aint_diff_ as pmpi_aint_diff_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));

#else
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *);

#if defined(F77_NAME_UPPER)
#pragma weak mpi_aint_diff__ = MPI_AINT_DIFF
#pragma weak mpi_aint_diff_ = MPI_AINT_DIFF
#pragma weak mpi_aint_diff = MPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_AINT_DIFF = mpi_aint_diff__
#pragma weak mpi_aint_diff_ = mpi_aint_diff__
#pragma weak mpi_aint_diff = mpi_aint_diff__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_AINT_DIFF = mpi_aint_diff_
#pragma weak mpi_aint_diff__ = mpi_aint_diff_
#pragma weak mpi_aint_diff = mpi_aint_diff_
#else
#pragma weak MPI_AINT_DIFF = mpi_aint_diff
#pragma weak mpi_aint_diff__ = mpi_aint_diff
#pragma weak mpi_aint_diff_ = mpi_aint_diff
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("MPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("MPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("MPI_AINT_DIFF")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *);
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff_")));

#else
extern FORT_DLL_SPEC MPI_Aint FORT_CALL MPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("mpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff(MPI_Aint *, MPI_Aint *);

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC MPI_Aint FORT_CALL PMPI_AINT_DIFF(MPI_Aint *, MPI_Aint *);
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff__(MPI_Aint *, MPI_Aint *);
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff_(MPI_Aint *, MPI_Aint *);
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff(MPI_Aint *, MPI_Aint *);

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_aint_diff__ = PMPI_AINT_DIFF
#pragma weak pmpi_aint_diff_ = PMPI_AINT_DIFF
#pragma weak pmpi_aint_diff = PMPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_AINT_DIFF = pmpi_aint_diff__
#pragma weak pmpi_aint_diff_ = pmpi_aint_diff__
#pragma weak pmpi_aint_diff = pmpi_aint_diff__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_AINT_DIFF = pmpi_aint_diff_
#pragma weak pmpi_aint_diff__ = pmpi_aint_diff_
#pragma weak pmpi_aint_diff = pmpi_aint_diff_
#else
#pragma weak PMPI_AINT_DIFF = pmpi_aint_diff
#pragma weak pmpi_aint_diff__ = pmpi_aint_diff
#pragma weak pmpi_aint_diff_ = pmpi_aint_diff
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("PMPI_AINT_DIFF")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL PMPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC MPI_Aint FORT_CALL PMPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff_")));

#else
extern FORT_DLL_SPEC MPI_Aint FORT_CALL PMPI_AINT_DIFF(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff__(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));
extern FORT_DLL_SPEC MPI_Aint FORT_CALL pmpi_aint_diff_(MPI_Aint *, MPI_Aint *) __attribute__((weak,alias("pmpi_aint_diff")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_aint_diff_ PMPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_aint_diff_ pmpi_aint_diff__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_aint_diff_ pmpi_aint_diff
#else
#define mpi_aint_diff_ pmpi_aint_diff_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Aint_diff
#define MPI_Aint_diff PMPI_Aint_diff 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_aint_diff_ MPI_AINT_DIFF
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_aint_diff_ mpi_aint_diff__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_aint_diff_ mpi_aint_diff
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC MPI_Aint FORT_CALL mpi_aint_diff_ (MPI_Aint *addr1, MPI_Aint *addr2)
{
    return MPI_Aint_diff(*addr1, *addr2);
}
