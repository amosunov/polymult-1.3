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

#ifndef INIT_H_
#define INIT_H_

#ifndef uint
#define uint unsigned int
#endif

#ifndef ulong
#define ulong unsigned long
#endif

#include <flint/nmod_poly.h>


// Block-by-block initialization

#define THETA3 0
void init_block_theta3(int * block, const ulong size, const ulong min);

#define THETA3_SQUARED 1
void init_block_theta3_squared(int * block, const ulong size, const ulong min);

#define NABLA 2
void init_block_nabla(int * block, const ulong size, const ulong min);

#define NABLA_SQUARED 3
void init_block_nabla_squared(int * block, const ulong size, const ulong min);

#define DOUBLE_NABLA 4
void init_block_double_nabla(int * block, const ulong size, const ulong min);

#define DOUBLE_NABLA_SQUARED 5
void init_block_double_nabla_squared(int * block, const ulong size, const ulong min);

#define THETA234 6
void init_block_theta234(int * block, const ulong size, const ulong min, const int mod);

#define ALPHA 7
void init_block_alpha(int * block, const ulong size, const ulong min, const int mod);

#define IS_FILE(X) (((ulong) X) > ALPHA)


// FLINT polynomials

void nmod_poly_theta3(nmod_poly_t poly, const ulong size);

void nmod_poly_theta234(nmod_poly_t poly, const ulong size);

void nmod_poly_to_files(const nmod_poly_t poly, const ulong size, const uint files, const char * resultname);

#endif /* INIT_H_ */
