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

#include "mult.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef uint
#define uint unsigned int
#endif

#ifndef ulong
#define ulong unsigned long
#endif

int main(int argc, char * argv[])
{
	if (argc != 10)
	{
		printf("Format: ./polymult [multiply/divide] [poly1] [poly2] [limit] [files] [bundle] [bound] [resultname] [folder]\n");
		printf("Values for poly1/poly2:\n");
		printf("0: theta3\t\t= 1 + 2q + 2q^4 + 2q^9 + ...\n");
		printf("1: theta3 squared\n");
		printf("2: nabla\t\t= 1 + q + q^3 + q^6 + q^10 + ...\n");
		printf("3: nabla squared\n");
		printf("4: double nabla\t= 1 + q^2 + q^6 + q^12 + ...\n");
		printf("5: double nabla squared\n");
		printf("6: theta2*theta3*theta4\t= 1 - 3q + 5q^3 - 7q^6 + 9q^10 - ...\n");
		printf("7: alpha series\n");
		exit(1);
	}

	const char type1 = atoi(argv[2]);
	const char type2 = atoi(argv[3]);
	const ulong limit = atol(argv[4]);
	const uint files = atoi(argv[5]);
	const uint bundle = atoi(argv[6]);
	const ulong bound = atol(argv[7]);
	const char * resultname = argv[8];
	const char * folder = argv[9];

	if ((limit / (files * bundle)) % (getpagesize() / (sizeof(ulong))) != 0)
	{
		printf("In order for MMAP to work, the quantity [limit]/([files] * [bundle]) = %lu should be divisible by the page size, which is %lu.\n", limit / (files * bundle), getpagesize() / sizeof(ulong));
		exit(1);
	}

	if (strcmp(argv[1], "divide") == 0)
	{
		divide(limit, files, bundle, bound, resultname, folder, type1, type2);
	}
	else
	{
		multiply(limit, files, bundle, bound, resultname, folder, type1, type2);
	}

	return 0;
}

