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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <flint/nmod_poly.h>

#include "init.h"

// Block-by-block initialization

void init_block_theta3(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i;

	if (min)
	{
		i = (ulong) ceil(sqrt(min));
	}
	else
	{
		block[0] = i = 1;
	}

	ulong t = (i * i - min);

	for (; t < size; i++)
	{
		block[t] = 2;
		t += (i << 1) + 1;
	}
}

void init_block_theta3_squared(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i = 0, x = 0, j, y, t;

	for (; x < min + size; i++)
	{
		j = (min > x) ? sqrt(min - x) : 0;

		y = j * j;

		if (min > x && y < min - x)
		{
			y += (j++ << 1) + 1;
		}

		t = x + y - min;

		for (; t < size; j++)
		{
			block[t] += (x ? 2 : 1) * (y ? 2 : 1);
			y += (j << 1) + 1;
			t += (j << 1) + 1;
		}

		x += (i << 1) + 1;
	}
}

void init_block_nabla(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i = (ulong) ceil((sqrt(1 + (min << 3)) + 1) / 2);

	ulong t = (i * (i - 1)/2 - min);

	for (; t < size; i++)
	{
		block[t] = 1;
		t += i;
	}
}

void init_block_nabla_squared(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i = 1, x = 0, j, y;

	ulong t;

	for (; x < min + size; i++)
	{
		j = (min > x) ? (sqrt(1 + ((min - x) << 3)) + 1) / 2 : 1;

		y = j * (j - 1) / 2;

		if (x < min && x + y < min)
		{
			y += j++;
		}

		t = (ulong) (x + y - min);

		for (; t < size; j++)
		{
			block[t]++;
			y += j;
			t += j;
		}

		x += i;
	}
}

void init_block_double_nabla(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i = (ulong) ceil((sqrt(1 + (min << 2)) + 1) / 2);

	ulong t = (i * (i - 1) - min);

	for (; t < size; i++)
	{
		block[t] = 1;
		t += (i << 1);
	}
}

void init_block_double_nabla_squared(int * block, const ulong size, const ulong min)
{
	memset(block, 0, size * sizeof(int));

	ulong i = 1, x = 0, j, y, t;

	for (; x < min + size; i++)
	{
		j = (min > x) ? (sqrt(1 + ((min - x) << 2)) + 1) / 2 : 1;

		y = j * (j - 1);

		if (min > x && y < min - x)
		{
			y += (j << 1);
			j++;
		}

		t = (x + y - min);

		for (; t < size; j++)
		{
			block[t]++;
			y += (j << 1);
			t += (j << 1);
		}

		x += (i << 1);
	}
}

void init_block_theta234(int * block, const ulong size, const ulong min, const int mod)
{
	memset(block, 0, size * sizeof(int));

	ulong i = (ulong) ceil((sqrt(1 + (min << 3)) - 1) / 2);

	ulong t;

	const ulong half = mod >> 1;

	for (t = ((i * (i + 1)) >> 1) - min; t < size; i++)
	{
		if (i & 1)
		{
			block[t] = - (i << 1) - 1;

			if (mod)
			{
				block[t] += ((-block[t]) / mod + 1) * mod;

				if (block[t] >= mod)
				{
					block[t] -= mod;
				}
			}
		}
		else
		{
			block[t] = (i << 1) + 1;
		}

		if (block[t] > half)
		{
			block[t] -= mod;
		}

		t += i + 1;
	}
}

void init_block_alpha(int * block, const ulong size, const ulong min, const int mod)
{
	memset(block, 0, sizeof(int) * size);

	ulong min0 = min + 1;

	long offset;

	ulong max = min0 + size;

	ulong u = sqrt((min0 + size) << 1);

	ulong j, k, m, d, sq_div, step;

	d = 2;
	m = 3;

	while (m < max && d <= u)
	{
		sq_div = d * d;

		step = d << 1;


		// k - even
		offset = m - min0;

		if (offset <= 0)
		{
			k = (-offset) / step;
			k = (k + (-offset != k * step)) << 1;
			offset += k * d;
		}

		for (j = offset; j < size; j += step)
		{
			block[j] -= sq_div;
		}


		// k - odd
		offset = m + d - min0;

		if (offset <= 0)
		{
			k = (-offset) / step;
			k = (k + (-offset != k * step)) << 1;
			offset += k * d;
		}

		for (j = offset; j < size; j += step)
		{
			block[j] += sq_div;
		}

		m += (d << 1) + 3;
		d += 2;
	}

	d = 1;
	m = 1;

	while (m < max && d <= u)
	{
		sq_div = d * d;

		step = d << 1;


		// k - even
		offset = m - min0;

		if (offset <= 0)
		{
			k = (-offset) / step;
			k = (k + (-offset != k * step)) << 1;
			offset += k * d;
		}

		for (j = offset; j < size; j += step)
		{
			block[j] += sq_div;
		}


		// k - odd
		offset = m + d - min0;

		if (offset <= 0)
		{
			k = (-offset) / step;
			k = (k + (-offset != k * step)) << 1;
			offset += k * d;
		}

		for (j = offset; j < size; j += step)
		{
			block[j] -= sq_div;
		}

		m += (d << 1) + 3;
		d += 2;
	}

	ulong t;

	if (mod)
	{
		int half = mod >> 1;

		for (t = 0; t < size; t++)
		{
			if (block[t] < 0)
			{
				block[t] += ((-block[t]) / mod + 1) * mod;

				if (block[t] >= mod)
				{
					block[t] -= mod;
				}
			}

			if (block[t] > half)
			{
				block[t] -= mod;
			}
		}
	}
}


// FLINT polynomials

void nmod_poly_theta3(nmod_poly_t poly, const ulong size)
{

	nmod_poly_set_coeff_ui(poly, 0, 1);

	nmod_poly_fit_length(poly, size);

	ulong i = 1;
	ulong sq_num = 1;

	while (sq_num <= size)
	{
		nmod_poly_set_coeff_ui(poly, sq_num, 2);
		sq_num += (i++ << 1) + 1;
	}
}

void nmod_poly_theta234(nmod_poly_t poly, const ulong size)
{
	ulong modulus = nmod_poly_modulus(poly);

	const int max_threads = omp_get_max_threads();

	const ulong max = floor((sqrt(1 + (size << 3)) - 1) / 2);

	nmod_poly_fit_length(poly, size);

	poly->length = ((max * (max + 1)) >> 1) + 1;

	#pragma omp parallel for
	for (int n = 0; n < max_threads; n++)
	{
		size_t loc_size = size / max_threads;

		if (n == max_threads - 1)
		{
			loc_size += size % max_threads;
		}

		memset(poly->coeffs + n * (size / max_threads), 0, sizeof(ulong) * loc_size);

		const ulong max = (n == max_threads - 1) ? ((n + 1) * (size / max_threads)) : size;

		ulong i = ceil((sqrt(1 + (n * (size / max_threads) << 3)) - 1) / 2);

		long coeff;

		for (ulong t = (i * (i + 1)) >> 1; t < max; i++)
		{
			if (i & 1)
			{
				coeff = - (i << 1) - 1;

				coeff += ((-coeff) / modulus + 1) * modulus;
			}
			else
			{
				coeff = (i << 1) + 1;
			}

			poly->coeffs[t] = coeff;

			t += i + 1;
		}
	}
}

void nmod_poly_to_files(const nmod_poly_t poly, const ulong size, const uint files, const char * resultname)
{
	const ulong blocksize = size / files;

	const ulong mod = nmod_poly_modulus(poly);

	const ulong half = mod >> 1;

	//#pragma omp parallel
	{
		int * coeff = malloc(sizeof(int) * blocksize);

		char name[150];

		//#pragma omp for
		for (uint s = 0; s < files; s++)
		{
			ulong cur = s * blocksize;

			for (ulong j = 0; j < blocksize; j++)
			{
				coeff[j] = (int) nmod_poly_get_coeff_ui(poly, cur++);

				if (coeff[j] > half)
				{
					coeff[j] -= mod;
				}
			}

			sprintf(name, "%s%u", resultname, s);

			int fd = open(name, O_RDWR | O_CREAT, 0644);

			if (fd == -1)
			{
				perror("In nmod_poly_to_files: unable to open a file for reading.\n");
				exit(1);
			}

			ftruncate(fd, sizeof(int) * blocksize);

			int * file = (int *) mmap(0, sizeof(int) * blocksize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

			if (file == MAP_FAILED)
			{
				perror("In nmod_poly_to_files: mapping failed when writing.\n");
				exit(1);
			}

			memcpy(file, coeff, sizeof(int) * blocksize);

			munmap(file, sizeof(int) * blocksize);

			close(fd);
		}

		free(coeff);
	}
}
