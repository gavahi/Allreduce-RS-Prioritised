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
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_INFO_GET_VALUELEN = PMPI_INFO_GET_VALUELEN
#pragma weak mpi_info_get_valuelen__ = PMPI_INFO_GET_VALUELEN
#pragma weak mpi_info_get_valuelen_ = PMPI_INFO_GET_VALUELEN
#pragma weak mpi_info_get_valuelen = PMPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen__
#pragma weak mpi_info_get_valuelen__ = pmpi_info_get_valuelen__
#pragma weak mpi_info_get_valuelen_ = pmpi_info_get_valuelen__
#pragma weak mpi_info_get_valuelen = pmpi_info_get_valuelen__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen_
#pragma weak mpi_info_get_valuelen__ = pmpi_info_get_valuelen_
#pragma weak mpi_info_get_valuelen_ = pmpi_info_get_valuelen_
#pragma weak mpi_info_get_valuelen = pmpi_info_get_valuelen_
#else
#pragma weak MPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen
#pragma weak mpi_info_get_valuelen__ = pmpi_info_get_valuelen
#pragma weak mpi_info_get_valuelen_ = pmpi_info_get_valuelen
#pragma weak mpi_info_get_valuelen = pmpi_info_get_valuelen
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak MPI_INFO_GET_VALUELEN = PMPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_info_get_valuelen__ = pmpi_info_get_valuelen__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_info_get_valuelen = pmpi_info_get_valuelen
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_info_get_valuelen_ = pmpi_info_get_valuelen_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_INFO_GET_VALUELEN  MPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_info_get_valuelen__  mpi_info_get_valuelen__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_info_get_valuelen  mpi_info_get_valuelen
#else
#pragma _HP_SECONDARY_DEF pmpi_info_get_valuelen_  mpi_info_get_valuelen_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_INFO_GET_VALUELEN as PMPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_info_get_valuelen__ as pmpi_info_get_valuelen__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_info_get_valuelen as pmpi_info_get_valuelen
#else
#pragma _CRI duplicate mpi_info_get_valuelen_ as pmpi_info_get_valuelen_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_info_get_valuelen__ = MPI_INFO_GET_VALUELEN
#pragma weak mpi_info_get_valuelen_ = MPI_INFO_GET_VALUELEN
#pragma weak mpi_info_get_valuelen = MPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INFO_GET_VALUELEN = mpi_info_get_valuelen__
#pragma weak mpi_info_get_valuelen_ = mpi_info_get_valuelen__
#pragma weak mpi_info_get_valuelen = mpi_info_get_valuelen__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INFO_GET_VALUELEN = mpi_info_get_valuelen_
#pragma weak mpi_info_get_valuelen__ = mpi_info_get_valuelen_
#pragma weak mpi_info_get_valuelen = mpi_info_get_valuelen_
#else
#pragma weak MPI_INFO_GET_VALUELEN = mpi_info_get_valuelen
#pragma weak mpi_info_get_valuelen__ = mpi_info_get_valuelen
#pragma weak mpi_info_get_valuelen_ = mpi_info_get_valuelen
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_INFO_GET_VALUELEN")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_info_get_valuelen__ = PMPI_INFO_GET_VALUELEN
#pragma weak pmpi_info_get_valuelen_ = PMPI_INFO_GET_VALUELEN
#pragma weak pmpi_info_get_valuelen = PMPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen__
#pragma weak pmpi_info_get_valuelen_ = pmpi_info_get_valuelen__
#pragma weak pmpi_info_get_valuelen = pmpi_info_get_valuelen__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen_
#pragma weak pmpi_info_get_valuelen__ = pmpi_info_get_valuelen_
#pragma weak pmpi_info_get_valuelen = pmpi_info_get_valuelen_
#else
#pragma weak PMPI_INFO_GET_VALUELEN = pmpi_info_get_valuelen
#pragma weak pmpi_info_get_valuelen__ = pmpi_info_get_valuelen
#pragma weak pmpi_info_get_valuelen_ = pmpi_info_get_valuelen
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_INFO_GET_VALUELEN")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_INFO_GET_VALUELEN( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen__( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_info_get_valuelen_( MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint *, MPI_Fint *, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_info_get_valuelen")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_info_get_valuelen_ PMPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_info_get_valuelen_ pmpi_info_get_valuelen__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_info_get_valuelen_ pmpi_info_get_valuelen
#else
#define mpi_info_get_valuelen_ pmpi_info_get_valuelen_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Info_get_valuelen
#define MPI_Info_get_valuelen PMPI_Info_get_valuelen 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_info_get_valuelen_ MPI_INFO_GET_VALUELEN
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_info_get_valuelen_ mpi_info_get_valuelen__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_info_get_valuelen_ mpi_info_get_valuelen
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_info_get_valuelen_ ( MPI_Fint *v1, char *v2 FORT_MIXED_LEN(d2), MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *ierr FORT_END_LEN(d2) ){
    char *p2;
    int l4;

    {char *p = v2 + d2 - 1;
     int  li;
        while (*p == ' ' && p > v2) p--;
        p++;
        p2 = (char *)malloc(p-v2 + 1);
        for (li=0; li<(p-v2); li++) { p2[li] = v2[li]; }
        p2[li] = 0; 
    }
    *ierr = MPI_Info_get_valuelen( (MPI_Info)(*v1), p2, v3, &l4 );
    if (*ierr == MPI_SUCCESS) *v4 = MPII_TO_FLOG(l4);
    free( p2 );
}
