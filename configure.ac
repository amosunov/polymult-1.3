#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT(polymult, 1.3, amosunov@uwaterloo.ca)
AM_INIT_AUTOMAKE(polymult, 1.3)
AC_CONFIG_SRCDIR([init.c])
AC_CONFIG_HEADERS([config.h])
AC_PREFIX_DEFAULT (/usr/local)

# Checks for programs.
AC_PROG_CC

# Checks for libraries.
AC_PROG_RANLIB

# Checks for header files.
AC_CHECK_HEADERS([stdio.h fcntl.h stdlib.h string.h sys/statvfs.h sys/time.h unistd.h])
AC_CHECK_HEADER(omp.h, ,
  [AC_MSG_ERROR([OpenMP not found, see http://openmp.org/])],[[#if HAVE_OMP_H
# include <omp.h>
#endif
]])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_OFF_T
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([floor ftruncate getpagesize gettimeofday memset mkdir munmap sqrt])
AC_CHECK_LIB(gmp, __gmpz_init, ,
  [AC_MSG_ERROR([GNU MP not found, see https://gmplib.org/])])
AC_CHECK_LIB(flint, flint_malloc, ,
  [AC_MSG_ERROR([FLINT not found, see http://flintlib.org/])])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
