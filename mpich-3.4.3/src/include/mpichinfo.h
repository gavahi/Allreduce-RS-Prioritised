/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* This file creates strings for the most important configuration options.
   These are then used in the file src/mpi/init/initthread.c to initialize
   global variables that will then be included in both the library and
   executables, providing a way to determine what version and features of
   MPICH were used with a particular library or executable.
*/
#ifndef MPICHINFO_H_INCLUDED
#define MPICHINFO_H_INCLUDED

#define MPICH_CONFIGURE_ARGS_CLEAN "--prefix=/home/gavahi/allreduce-RS-Prioritised2/mpich-3.4.3/Install_mpich --with-device=ch3:nemesis"
#define MPICH_VERSION_DATE "Thu Dec 16 11:20:57 CST 2021"
#define MPICH_DEVICE "ch3:nemesis"
#define MPICH_COMPILER_CC "gcc -std=gnu99 -std=gnu99    -O2"
#define MPICH_COMPILER_CXX "g++   -O2"
#define MPICH_COMPILER_F77 "gfortran   -O2"
#define MPICH_COMPILER_FC "gfortran   -O2"
#define MPICH_CUSTOM_STRING ""
#define MPICH_ABIVERSION "13:12:1"

#endif
