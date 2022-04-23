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
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_CART_RANK = PMPI_CART_RANK
#pragma weak mpi_cart_rank__ = PMPI_CART_RANK
#pragma weak mpi_cart_rank_ = PMPI_CART_RANK
#pragma weak mpi_cart_rank = PMPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_CART_RANK = pmpi_cart_rank__
#pragma weak mpi_cart_rank__ = pmpi_cart_rank__
#pragma weak mpi_cart_rank_ = pmpi_cart_rank__
#pragma weak mpi_cart_rank = pmpi_cart_rank__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_CART_RANK = pmpi_cart_rank_
#pragma weak mpi_cart_rank__ = pmpi_cart_rank_
#pragma weak mpi_cart_rank_ = pmpi_cart_rank_
#pragma weak mpi_cart_rank = pmpi_cart_rank_
#else
#pragma weak MPI_CART_RANK = pmpi_cart_rank
#pragma weak mpi_cart_rank__ = pmpi_cart_rank
#pragma weak mpi_cart_rank_ = pmpi_cart_rank
#pragma weak mpi_cart_rank = pmpi_cart_rank
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#pragma weak MPI_CART_RANK = PMPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_rank__ = pmpi_cart_rank__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_rank = pmpi_cart_rank
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#pragma weak mpi_cart_rank_ = pmpi_cart_rank_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_CART_RANK  MPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_cart_rank__  mpi_cart_rank__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_cart_rank  mpi_cart_rank
#else
#pragma _HP_SECONDARY_DEF pmpi_cart_rank_  mpi_cart_rank_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_CART_RANK as PMPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_cart_rank__ as pmpi_cart_rank__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_cart_rank as pmpi_cart_rank
#else
#pragma _CRI duplicate mpi_cart_rank_ as pmpi_cart_rank_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_cart_rank__ = MPI_CART_RANK
#pragma weak mpi_cart_rank_ = MPI_CART_RANK
#pragma weak mpi_cart_rank = MPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_CART_RANK = mpi_cart_rank__
#pragma weak mpi_cart_rank_ = mpi_cart_rank__
#pragma weak mpi_cart_rank = mpi_cart_rank__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_CART_RANK = mpi_cart_rank_
#pragma weak mpi_cart_rank__ = mpi_cart_rank_
#pragma weak mpi_cart_rank = mpi_cart_rank_
#else
#pragma weak MPI_CART_RANK = mpi_cart_rank
#pragma weak mpi_cart_rank__ = mpi_cart_rank
#pragma weak mpi_cart_rank_ = mpi_cart_rank
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_CART_RANK")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL mpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_cart_rank__ = PMPI_CART_RANK
#pragma weak pmpi_cart_rank_ = PMPI_CART_RANK
#pragma weak pmpi_cart_rank = PMPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_CART_RANK = pmpi_cart_rank__
#pragma weak pmpi_cart_rank_ = pmpi_cart_rank__
#pragma weak pmpi_cart_rank = pmpi_cart_rank__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_CART_RANK = pmpi_cart_rank_
#pragma weak pmpi_cart_rank__ = pmpi_cart_rank_
#pragma weak pmpi_cart_rank = pmpi_cart_rank_
#else
#pragma weak PMPI_CART_RANK = pmpi_cart_rank
#pragma weak pmpi_cart_rank__ = pmpi_cart_rank
#pragma weak pmpi_cart_rank_ = pmpi_cart_rank
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_CART_RANK")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_CART_RANK( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank__( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_cart_rank_( MPI_Fint *, MPI_Fint [], MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_cart_rank")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_cart_rank_ PMPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_cart_rank_ pmpi_cart_rank__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_cart_rank_ pmpi_cart_rank
#else
#define mpi_cart_rank_ pmpi_cart_rank_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Cart_rank
#define MPI_Cart_rank PMPI_Cart_rank 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_cart_rank_ MPI_CART_RANK
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_cart_rank_ mpi_cart_rank__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_cart_rank_ mpi_cart_rank
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_cart_rank_ ( MPI_Fint *v1, MPI_Fint v2[], MPI_Fint *v3, MPI_Fint *ierr ){
    *ierr = MPI_Cart_rank( (MPI_Comm)(*v1), v2, v3 );
}
