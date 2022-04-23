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
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak MPI_FILE_GET_VIEW = PMPI_FILE_GET_VIEW
#pragma weak mpi_file_get_view__ = PMPI_FILE_GET_VIEW
#pragma weak mpi_file_get_view_ = PMPI_FILE_GET_VIEW
#pragma weak mpi_file_get_view = PMPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_GET_VIEW = pmpi_file_get_view__
#pragma weak mpi_file_get_view__ = pmpi_file_get_view__
#pragma weak mpi_file_get_view_ = pmpi_file_get_view__
#pragma weak mpi_file_get_view = pmpi_file_get_view__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_GET_VIEW = pmpi_file_get_view_
#pragma weak mpi_file_get_view__ = pmpi_file_get_view_
#pragma weak mpi_file_get_view_ = pmpi_file_get_view_
#pragma weak mpi_file_get_view = pmpi_file_get_view_
#else
#pragma weak MPI_FILE_GET_VIEW = pmpi_file_get_view
#pragma weak mpi_file_get_view__ = pmpi_file_get_view
#pragma weak mpi_file_get_view_ = pmpi_file_get_view
#pragma weak mpi_file_get_view = pmpi_file_get_view
#endif



#elif defined(HAVE_PRAGMA_WEAK)

#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak MPI_FILE_GET_VIEW = PMPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_file_get_view__ = pmpi_file_get_view__
#elif !defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_file_get_view = pmpi_file_get_view
#else
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#pragma weak mpi_file_get_view_ = pmpi_file_get_view_
#endif

#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#if defined(F77_NAME_UPPER)
#pragma _HP_SECONDARY_DEF PMPI_FILE_GET_VIEW  MPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_get_view__  mpi_file_get_view__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _HP_SECONDARY_DEF pmpi_file_get_view  mpi_file_get_view
#else
#pragma _HP_SECONDARY_DEF pmpi_file_get_view_  mpi_file_get_view_
#endif

#elif defined(HAVE_PRAGMA_CRI_DUP)
#if defined(F77_NAME_UPPER)
#pragma _CRI duplicate MPI_FILE_GET_VIEW as PMPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma _CRI duplicate mpi_file_get_view__ as pmpi_file_get_view__
#elif !defined(F77_NAME_LOWER_USCORE)
#pragma _CRI duplicate mpi_file_get_view as pmpi_file_get_view
#else
#pragma _CRI duplicate mpi_file_get_view_ as pmpi_file_get_view_
#endif

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));

#endif
#endif /* HAVE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */
/* End MPI profiling block */


/* These definitions are used only for generating the Fortran wrappers */
#if defined(USE_WEAK_SYMBOLS) && defined(USE_ONLY_MPI_NAMES)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#if defined(F77_NAME_UPPER)
#pragma weak mpi_file_get_view__ = MPI_FILE_GET_VIEW
#pragma weak mpi_file_get_view_ = MPI_FILE_GET_VIEW
#pragma weak mpi_file_get_view = MPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak MPI_FILE_GET_VIEW = mpi_file_get_view__
#pragma weak mpi_file_get_view_ = mpi_file_get_view__
#pragma weak mpi_file_get_view = mpi_file_get_view__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak MPI_FILE_GET_VIEW = mpi_file_get_view_
#pragma weak mpi_file_get_view__ = mpi_file_get_view_
#pragma weak mpi_file_get_view = mpi_file_get_view_
#else
#pragma weak MPI_FILE_GET_VIEW = mpi_file_get_view
#pragma weak mpi_file_get_view__ = mpi_file_get_view
#pragma weak mpi_file_get_view_ = mpi_file_get_view
#endif
#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("MPI_FILE_GET_VIEW")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view_")));

#else
extern FORT_DLL_SPEC void FORT_CALL MPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("mpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL mpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#endif
#endif

#endif

/* Map the name to the correct form */
#ifndef MPICH_MPI_FROM_PMPI
#if defined(USE_WEAK_SYMBOLS)
#if defined(HAVE_MULTIPLE_PRAGMA_WEAK)
/* Define the weak versions of the PMPI routine*/
#ifndef F77_NAME_UPPER
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_2USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER_USCORE
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );
#endif
#ifndef F77_NAME_LOWER
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL );

#endif

#if defined(F77_NAME_UPPER)
#pragma weak pmpi_file_get_view__ = PMPI_FILE_GET_VIEW
#pragma weak pmpi_file_get_view_ = PMPI_FILE_GET_VIEW
#pragma weak pmpi_file_get_view = PMPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#pragma weak PMPI_FILE_GET_VIEW = pmpi_file_get_view__
#pragma weak pmpi_file_get_view_ = pmpi_file_get_view__
#pragma weak pmpi_file_get_view = pmpi_file_get_view__
#elif defined(F77_NAME_LOWER_USCORE)
#pragma weak PMPI_FILE_GET_VIEW = pmpi_file_get_view_
#pragma weak pmpi_file_get_view__ = pmpi_file_get_view_
#pragma weak pmpi_file_get_view = pmpi_file_get_view_
#else
#pragma weak PMPI_FILE_GET_VIEW = pmpi_file_get_view
#pragma weak pmpi_file_get_view__ = pmpi_file_get_view
#pragma weak pmpi_file_get_view_ = pmpi_file_get_view
#endif /* Test on name mapping */

#elif defined(HAVE_WEAK_ATTRIBUTE)
#if defined(F77_NAME_UPPER)
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("PMPI_FILE_GET_VIEW")));

#elif defined(F77_NAME_LOWER_2USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view__")));

#elif defined(F77_NAME_LOWER_USCORE)
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view_")));

#else
extern FORT_DLL_SPEC void FORT_CALL PMPI_FILE_GET_VIEW( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view__( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));
extern FORT_DLL_SPEC void FORT_CALL pmpi_file_get_view_( MPI_Fint *, MPI_Offset*, MPI_Fint *, MPI_Fint *, char * FORT_MIXED_LEN_DECL, MPI_Fint * FORT_END_LEN_DECL ) __attribute__((weak,alias("pmpi_file_get_view")));

#endif /* Test on name mapping */
#endif /* HAVE_MULTIPLE_PRAGMA_WEAK */
#endif /* USE_WEAK_SYMBOLS */

#ifdef F77_NAME_UPPER
#define mpi_file_get_view_ PMPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_get_view_ pmpi_file_get_view__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_get_view_ pmpi_file_get_view
#else
#define mpi_file_get_view_ pmpi_file_get_view_
#endif /* Test on name mapping */

#ifdef F77_USE_PMPI
/* This defines the routine that we call, which must be the PMPI version
   since we're renaming the Fortran entry as the pmpi version.  The MPI name
   must be undefined first to prevent any conflicts with previous renamings. */
#undef MPI_File_get_view
#define MPI_File_get_view PMPI_File_get_view 
#endif

#else

#ifdef F77_NAME_UPPER
#define mpi_file_get_view_ MPI_FILE_GET_VIEW
#elif defined(F77_NAME_LOWER_2USCORE)
#define mpi_file_get_view_ mpi_file_get_view__
#elif !defined(F77_NAME_LOWER_USCORE)
#define mpi_file_get_view_ mpi_file_get_view
/* Else leave name alone */
#endif


#endif /* MPICH_MPI_FROM_PMPI */

/* Prototypes for the Fortran interfaces */
#include "fproto.h"
FORT_DLL_SPEC void FORT_CALL mpi_file_get_view_ ( MPI_Fint *v1, MPI_Offset*v2, MPI_Fint *v3, MPI_Fint *v4, char *v5 FORT_MIXED_LEN(d5), MPI_Fint *ierr FORT_END_LEN(d5) ){
#ifdef MPI_MODE_RDONLY
    char *p5;
    p5 = (char *)MPIU_Malloc( d5 + 1 );
    *ierr = MPI_File_get_view( MPI_File_f2c(*v1), v2, (MPI_Datatype *)(v3), (MPI_Datatype *)(v4), p5 );

    if (!*ierr) {char *p = v5, *pc=p5;
        while (*pc) {*p++ = *pc++;}
        while ((p-v5) < d5) { *p++ = ' '; }
    }
    MPIU_Free( p5 );
#else
*ierr = MPI_ERR_INTERN;
#endif
}
