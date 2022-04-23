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
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_OP_CREATE = PMPI_OP_CREATE
#pragma weak mpi_op_create__ = PMPI_OP_CREATE
#pragma weak mpi_op_create_ = PMPI_OP_CREATE
#pragma weak mpi_op_create = PMPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_OP_CREATE = pmpi_op_create__
#pragma weak mpi_op_create__ = pmpi_op_create__
#pragma weak mpi_op_create_ = pmpi_op_create__
#pragma weak mpi_op_create = pmpi_op_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_OP_CREATE = pmpi_op_create_
#pragma weak mpi_op_create__ = pmpi_op_create_
#pragma weak mpi_op_create_ = pmpi_op_create_
#pragma weak mpi_op_create = pmpi_op_create_
#else
#pragma weak MPI_OP_CREATE = pmpi_op_create
#pragma weak mpi_op_create__ = pmpi_op_create
#pragma weak mpi_op_create_ = pmpi_op_create
#pragma weak mpi_op_create = pmpi_op_create
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak MPI_OP_CREATE = PMPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_op_create__ = pmpi_op_create__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_op_create = pmpi_op_create
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#pragma weak mpi_op_create_ = pmpi_op_create_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_OP_CREATE  MPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_op_create__  mpi_op_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_op_create  mpi_op_create
#else
#pragma _HP_SECONDARY_DEF pmpi_op_create_  mpi_op_create_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_OP_CREATE as PMPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_op_create__ as pmpi_op_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_op_create as pmpi_op_create
#else
#pragma _CRI duplicate mpi_op_create_ as pmpi_op_create_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_op_create__ = MPI_OP_CREATE
#pragma weak mpi_op_create_ = MPI_OP_CREATE
#pragma weak mpi_op_create = MPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_OP_CREATE = mpi_op_create__
#pragma weak mpi_op_create_ = mpi_op_create__
#pragma weak mpi_op_create = mpi_op_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_OP_CREATE = mpi_op_create_
#pragma weak mpi_op_create__ = mpi_op_create_
#pragma weak mpi_op_create = mpi_op_create_
#else
#pragma weak MPI_OP_CREATE = mpi_op_create
#pragma weak mpi_op_create__ = mpi_op_create
#pragma weak mpi_op_create_ = mpi_op_create
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("MPI_OP_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("mpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL mpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_op_create__ = PMPI_OP_CREATE
#pragma weak pmpi_op_create_ = PMPI_OP_CREATE
#pragma weak pmpi_op_create = PMPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_OP_CREATE = pmpi_op_create__
#pragma weak pmpi_op_create_ = pmpi_op_create__
#pragma weak pmpi_op_create = pmpi_op_create__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_OP_CREATE = pmpi_op_create_
#pragma weak pmpi_op_create__ = pmpi_op_create_
#pragma weak pmpi_op_create = pmpi_op_create_
#else
#pragma weak PMPI_OP_CREATE = pmpi_op_create
#pragma weak pmpi_op_create__ = pmpi_op_create
#pragma weak pmpi_op_create_ = pmpi_op_create
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("PMPI_OP_CREATE")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_OP_CREATE( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create__( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_op_create_( MPI_User_function*, MPI_Fint *, MPI_Fint *, MPI_Fint * ) __attribute__((weak,alias("pmpi_op_create")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_op_create_ PMPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_op_create_ pmpi_op_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_op_create_ pmpi_op_create
#else
#define mpi_op_create_ pmpi_op_create_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_Op_create
#define MPI_Op_create PMPI_Op_create 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_op_create_ MPI_OP_CREATE
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_op_create_ mpi_op_create__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_op_create_ mpi_op_create
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_op_create_ ( MPI_User_function*v1, MPI_Fint *v2, MPI_Fint *v3, MPI_Fint *ierr ){
    int l2;
    l2 = MPII_FROM_FLOG(*v2);
    *ierr = MPI_Op_create( v1, l2, v3 );
}
