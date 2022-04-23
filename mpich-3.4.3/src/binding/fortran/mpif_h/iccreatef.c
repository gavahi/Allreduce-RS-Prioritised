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
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_INTERCOMM_CREATE = PMPI_INTERCOMM_CREATE
#pragma weak mpi_intercomm_create__ = PMPI_INTERCOMM_CREATE
#pragma weak mpi_intercomm_create_ = PMPI_INTERCOMM_CREATE
#pragma weak mpi_intercomm_create = PMPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INTERCOMM_CREATE = pmpi_intercomm_create__
#pragma weak mpi_intercomm_create__ = pmpi_intercomm_create__
#pragma weak mpi_intercomm_create_ = pmpi_intercomm_create__
#pragma weak mpi_intercomm_create = pmpi_intercomm_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INTERCOMM_CREATE = pmpi_intercomm_create_
#pragma weak mpi_intercomm_create__ = pmpi_intercomm_create_
#pragma weak mpi_intercomm_create_ = pmpi_intercomm_create_
#pragma weak mpi_intercomm_create = pmpi_intercomm_create_
#else
#pragma weak MPI_INTERCOMM_CREATE = pmpi_intercomm_create
#pragma weak mpi_intercomm_create__ = pmpi_intercomm_create
#pragma weak mpi_intercomm_create_ = pmpi_intercomm_create
#pragma weak mpi_intercomm_create = pmpi_intercomm_create
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_INTERCOMM_CREATE = PMPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_intercomm_create__ = pmpi_intercomm_create__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_intercomm_create = pmpi_intercomm_create
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_intercomm_create_ = pmpi_intercomm_create_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_INTERCOMM_CREATE  MPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_intercomm_create__  mpi_intercomm_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_intercomm_create  mpi_intercomm_create
#else
#pragma _HP_SECONDARY_DEF pmpi_intercomm_create_  mpi_intercomm_create_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_INTERCOMM_CREATE as PMPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_intercomm_create__ as pmpi_intercomm_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_intercomm_create as pmpi_intercomm_create
#else
#pragma _CRI duplicate mpi_intercomm_create_ as pmpi_intercomm_create_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_intercomm_create__ = MPI_INTERCOMM_CREATE
#pragma weak mpi_intercomm_create_ = MPI_INTERCOMM_CREATE
#pragma weak mpi_intercomm_create = MPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_INTERCOMM_CREATE = mpi_intercomm_create__
#pragma weak mpi_intercomm_create_ = mpi_intercomm_create__
#pragma weak mpi_intercomm_create = mpi_intercomm_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_INTERCOMM_CREATE = mpi_intercomm_create_
#pragma weak mpi_intercomm_create__ = mpi_intercomm_create_
#pragma weak mpi_intercomm_create = mpi_intercomm_create_
#else
#pragma weak MPI_INTERCOMM_CREATE = mpi_intercomm_create
#pragma weak mpi_intercomm_create__ = mpi_intercomm_create
#pragma weak mpi_intercomm_create_ = mpi_intercomm_create
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_INTERCOMM_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_intercomm_create__ = PMPI_INTERCOMM_CREATE
#pragma weak pmpi_intercomm_create_ = PMPI_INTERCOMM_CREATE
#pragma weak pmpi_intercomm_create = PMPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_INTERCOMM_CREATE = pmpi_intercomm_create__
#pragma weak pmpi_intercomm_create_ = pmpi_intercomm_create__
#pragma weak pmpi_intercomm_create = pmpi_intercomm_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_INTERCOMM_CREATE = pmpi_intercomm_create_
#pragma weak pmpi_intercomm_create__ = pmpi_intercomm_create_
#pragma weak pmpi_intercomm_create = pmpi_intercomm_create_
#else
#pragma weak PMPI_INTERCOMM_CREATE = pmpi_intercomm_create
#pragma weak pmpi_intercomm_create__ = pmpi_intercomm_create
#pragma weak pmpi_intercomm_create_ = pmpi_intercomm_create
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_INTERCOMM_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_INTERCOMM_CREATE( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create__( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_intercomm_create_( MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_intercomm_create")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_intercomm_create_ PMPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_intercomm_create_ pmpi_intercomm_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_intercomm_create_ pmpi_intercomm_create
#else
#define mpi_intercomm_create_ pmpi_intercomm_create_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Intercomm_create
#define MPI_Intercomm_create PMPI_Intercomm_create 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_intercomm_create_ MPI_INTERCOMM_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_intercomm_create_ mpi_intercomm_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_intercomm_create_ mpi_intercomm_create
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_intercomm_create_ ( MPI_Fint *v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *v4, MPI_Fint *v5, MPI_Fint *v6, MPI_Fint *ierr ){
    *ierr = MPI_Intercomm_create( (MPI_Comm)(*v1), (int)*v2, (MPI_Comm)(*v3), (int)*v4, (int)*v5, (MPI_Comm *)(v6) );
}
