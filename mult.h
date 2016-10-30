/*=============================================================================

    This file is part of POLYMULT.

    POLYMULT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    POLYMULT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with POLYMULT; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

=============================================================================*/
/******************************************************************************

    Copyright (C) 2016 Anton Mosunov
 
******************************************************************************/

#include <omp.h>

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/statvfs.h>

#include <flint/arith.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_vec.h>
#include <flint/ulong_extras.h>

#include "init.h"

// flags
#define NO_REMOVE 1
#define WITH_INVERSE 2
#define WITH_SQUARING 4

#define NO_TRUNCATE 1 // use in square() routine

#define MINPOW 25

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#ifndef uint
#define uint unsigned int
#endif

#ifndef ulong
#define ulong unsigned long
#endif

void init_primes(ulong * primes, const uint total_primes, const ulong lowerbound);

void init_files(const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle, const uint bitsize, const int mod,
		const char * resultname, const void * type, const char flags);

void ooc_multiply(const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle,
		const char * resultname, const char * name1, const char * name2);

void ooc_square(const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle,
		const char * resultname, const char * name1, const char flags);

// See the list of flags at the top of the file
void restore_coeff(const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle, const uint bitsize, const int mod,
		const char * resultname, const char * name1, const int flags);

void invert(const ulong * primes,
		const uint maxpow, const uint files, const uint bundle, const int mod,
		const char * resultname, const char * folder, const char type);

void multiply(const ulong limit, const uint files, const uint bundle, const ulong bound,
		const char * resultname, const char * folder, const char type1, const char type2);

void divide(const ulong limit, const uint files, const uint bundle, const ulong bound,
		const char * resultname, const char * folder, const char type1, const char type2);
