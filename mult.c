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

#include "mult.h"

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

#define INIT_PARAMS theta + n * (blocksize / max_threads), blocksize / max_threads + ((n == max_threads - 1) ? (blocksize % max_threads) : 0), s * blocksize + n * (blocksize / max_threads)

static int max_threads;

fmpz_comb_t comb;

void init_primes(ulong * primes, const uint total_primes, const ulong lowerbound)
{
	primes[0] = n_nextprime(lowerbound, 1);

	for (uint i = 1; i < total_primes; i++)
	{
		primes[i] = n_nextprime(primes[i - 1], 1);
	}
}


//0 - theta3
//1 - theta3 squared
//2 - theta234
//3 - alpha series
void init_files(const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle, const uint bitsize, const int mod,
		const char * resultname, const void * type, const char flags)
{
	const ulong blocksize = limit / files;
	const ulong bundlesize = blocksize / bundle;

	char name[300];

	struct timeval begin, end;
	ulong exec_time, init_time = 0, bund_time = 0, red_time = 0, save_time = 0;

	int fd0, fd1;

	int * theta = (int *) valloc(sizeof(int) * blocksize);
	fmpz * c = _fmpz_vec_init(bundlesize);

	ulong * residues[total_primes];

	for (int k = 0; k < total_primes; k++)
	{
		residues[k] = (ulong *) malloc(sizeof(ulong) * bundlesize);
	}

	fmpz_comb_temp_t comb_temp[max_threads];
	ulong * temp_residues[max_threads];

	for (int n = 0; n < max_threads; n++)
	{
		fmpz_comb_temp_init(comb_temp[n], comb);
		temp_residues[n] = (ulong *) malloc(sizeof(ulong) * total_primes);
	}


	for (uint s = 0; s < files; s++)
	{
		#ifdef DEBUG
		printf("File %u of %u\n", s, files - 1);
		#endif

		gettimeofday(&begin, NULL);
		if (!IS_FILE(type))
		{
			#pragma omp parallel
			{
				int n = omp_get_thread_num();

				if (type == THETA3)
				{
					init_block_theta3(INIT_PARAMS);
				}
				else if (type == (void *) THETA3_SQUARED)
				{
					init_block_theta3_squared(INIT_PARAMS);
				}
				else if (type == (void *) NABLA)
				{
					init_block_nabla(INIT_PARAMS);
				}
				else if (type == (void *) NABLA_SQUARED)
				{
					init_block_nabla_squared(INIT_PARAMS);
				}
				else if (type == (void *) DOUBLE_NABLA)
				{
					init_block_double_nabla(INIT_PARAMS);
				}
				else if (type == (void *) DOUBLE_NABLA_SQUARED)
				{
					init_block_double_nabla_squared(INIT_PARAMS);
				}
				else if (type == (void *) THETA234)
				{
					init_block_theta234(INIT_PARAMS, mod);
				}
				else if (type == (void *) ALPHA)
				{
					init_block_alpha(INIT_PARAMS, mod);
				}
			}
		}
		else
		{
			sprintf(name, "%s%u", (char *) type, s);

			fd0 = open(name, O_RDONLY);

			if (fd0 == -1)
			{
				perror("In init_files: unable to open a file for reading.");
				exit(1);
			}

			read(fd0, theta, sizeof(int) * blocksize);

			close(fd0);

			#ifndef KEEP_FILES
			if (!((flags & NO_REMOVE) > 0))
			{
				remove(name);
			}
			#endif
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec * 1e6 + end.tv_usec) - (begin.tv_sec * 1e6 + begin.tv_usec);
		init_time += exec_time;
		#ifdef DEBUG
		printf("Init: %.3f\t%.3f\n", exec_time / 1000000.0f, init_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);
		#pragma omp parallel
		{
			fmpz_t temp;
			fmpz_init2(temp, (bitsize * bundle + (sizeof(ulong) << 3)) / (sizeof(ulong) << 3));

			ulong t;

			#pragma omp for
			for (long j = 0; j < bundlesize; j++)
			{
				int * a = theta + j * bundle;

				fmpz_set_si(c + j, a[0]);

				fmpz_one(temp);

				t = bitsize;

				for (uint r = 1; r < bundle; r++)
				{
					fmpz_set_si(temp, a[r]);

					fmpz_mul_2exp(temp, temp, t);

					fmpz_add(c + j, c + j, temp);

					t += bitsize;
				}

				a += bundle;
			}

			fmpz_clear(temp);
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec * 1e6 + end.tv_usec) - (begin.tv_sec * 1e6 + begin.tv_usec);
		bund_time += exec_time;
		#ifdef DEBUG
		printf("Bund: %.3f\t%.3f\n", exec_time / 1000000.0f, bund_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);
		#pragma omp parallel for
		for (int j = 0; j < bundlesize; j++)
		{
			const int n = omp_get_thread_num();

			fmpz_multi_mod_ui(temp_residues[n], c + j, comb, comb_temp[n]);

			for (int k = 0; k < total_primes; k++)
			{
				residues[k][j] = temp_residues[n][k];
			}
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		red_time += exec_time;
		#ifdef DEBUG
		printf("Redu: %.3f\t%.3f\n", exec_time / 1000000.0f, red_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);
		sprintf(name, "%s%u", resultname, s);

		fd1 = open(name, O_NDELAY | O_RDWR | O_CREAT, 0644);

		if (fd1 == -1)
		{
			printf(name);
			printf("\n");
			perror("In init_files: unable to open a file for writing.\n");
			exit(1);
		}

		ftruncate(fd1, sizeof(ulong) * total_primes * bundlesize);

		ulong * file = (ulong *) mmap(0, sizeof(ulong) * total_primes * bundlesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd1, 0);

		for (int k = total_primes - 1; k >= 0; k--)
		{
			memcpy(file + (total_primes - 1 - k) * bundlesize, residues[k], sizeof(ulong) * bundlesize);
		}

		munmap(file, sizeof(ulong) * total_primes * bundlesize);

		close(fd1);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		save_time += exec_time;
		#ifdef DEBUG
		printf("Save: %.3f\t%.3f\n", exec_time / 1000000.0f, save_time / 1000000.0f);
		#endif
	}

	free(theta);

	_fmpz_vec_clear(c, bundlesize);

	for (int k = 0; k < total_primes; k++)
	{
		free(residues[k]);
	}

	for (int n = 0; n < max_threads; n++)
	{
		fmpz_comb_temp_clear(comb_temp[n]);
		free(temp_residues[n]);
	}

	printf("Initialization took %.3f sec.\nBundling took %.3f sec.\nReduction took %.3f sec.\nSaving took %.3f sec.\n",
			init_time / 1000000.0f, bund_time / 1000000.0f, red_time / 1000000.0f, save_time / 1000000.0f);
}


void ooc_multiply(	const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle,
		const char * resultname, const char * name1, const char * name2)
{
	const ulong blocksize = limit / files;
	const ulong length = limit / bundle;
	const ulong bundlesize = blocksize / bundle;
	const ulong size = sizeof(ulong) * max_threads * bundlesize;
	const uint num_prime_blocks = total_primes / max_threads;

	struct timeval begin, end;
	ulong exec_time, init_time = 0, mult_time = 0, save_time = 0;

	nmod_poly_t fp[max_threads];
	nmod_poly_t gp[max_threads];

	#pragma omp parallel
	{
		const int n = omp_get_thread_num();
		nmod_poly_init(fp[n], 1);
		nmod_poly_fit_length(fp[n], length);
		nmod_poly_init(gp[n], 1);
		nmod_poly_fit_length(gp[n], length);
	}

	char name[300];
	const char * filename[2] = {name1, name2};
	nmod_poly_t * poly[2] = {fp, gp};

	for (uint i = 0; i < num_prime_blocks; i++)
	{
		#ifdef DEBUG
		printf("Block %u of %u\n", i, num_prime_blocks - 1);
		#endif

		gettimeofday(&begin, NULL);
		for (int l = 0; l < 2; l++)
		{
			nmod_poly_t * p = poly[l];

			sprintf(name, "%s%d", filename[l], 0);

			//#pragma omp parallel for
			for (int s = 0; s < files; s++)
			{
				char name[300];

				sprintf(name, "%s%d", filename[l], s);
				int fd = open(name, O_RDONLY);

				if (fd == -1)
				{
					perror("In ooc_multiply: unable to open a file for reading.\n");
					exit(1);
				}

				ulong * file = (ulong *) mmap(0, size, PROT_READ, MAP_SHARED, fd, size * (num_prime_blocks - 1 - i));

				if (file == MAP_FAILED)
				{
					close(fd);
					perror("In ooc_multiply: mapping failed when reading.\n");
					exit(1);
				}

				for (int n = 0; n < max_threads; n++)
				{
					memcpy(p[max_threads - 1 - n]->coeffs + s * bundlesize, file  + n * bundlesize, sizeof(ulong) * bundlesize);
				}

				munmap(file, size);

				#ifndef KEEP_FILES
				ftruncate(fd, size * (num_prime_blocks - 1 - i));
				#endif

				close(fd);
			}
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		init_time += exec_time;
		#ifdef DEBUG
		printf("Init: %.3f\t%.3f\n", exec_time / 1000000.0f, init_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);
		#pragma omp parallel
		{
			const int n = omp_get_thread_num();

			ulong prime = primes[i * max_threads + n];
			nmod_t mod;

			mod.n = prime;
			mod.ninv = n_preinvert_limb(prime);
			count_leading_zeros(mod.norm, prime);

			fp[n]->mod = mod;
			fp[n]->length = length;
			gp[n]->mod = mod;
			gp[n]->length = length;

			nmod_poly_mullow(fp[n], fp[n], gp[n], length);
			flint_cleanup();
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		mult_time += exec_time;
		#ifdef DEBUG
		printf("Mult: %.3f\t%.3f\n", exec_time / 1000000.0f, mult_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);
		#ifdef PARALLLEL_IO
		#pragma omp parallel for
		#endif
		for (int s = 0; s < files; s++)
		{
			char name[300];

			sprintf(name, "%s%d", resultname, s);
			int fd = open(name, O_RDWR | O_CREAT | O_APPEND, 0644);

			if (fd == -1)
			{
				perror("In ooc_multiply: unable to open a file for writing.\n");
				exit(1);
			}

			struct stat mystat;

			fstat(fd, &mystat);

			ftruncate(fd, mystat.st_size + size);

			ulong * file = (ulong *) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mystat.st_size);

			if (file == MAP_FAILED)
			{
				close(fd);
				perror("In ooc_multiply: mapping failed when writing.\n");
				exit(1);
			}

			for (int n = 0; n < max_threads; n++)
			{
				memcpy(file + n * bundlesize, fp[n]->coeffs + s * bundlesize, sizeof(ulong) * bundlesize);
			}

			munmap(file, size);

			close(fd);
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		save_time += exec_time;
		#ifdef DEBUG
		printf("Save: %.3f\t%.3f\n", exec_time / 1000000.0f, save_time / 1000000.0f);
		#endif
	}

	#pragma omp parallel
	{
		const int n = omp_get_thread_num();
		nmod_poly_clear(fp[n]);
		nmod_poly_clear(gp[n]);
	}

	#ifndef KEEP_FILES
	for (int s = 0; s < files; s++)
	{
		sprintf(name, "%s%d", name1, s);
		remove(name);

		sprintf(name, "%s%d", name2, s);
		remove(name);
	}
	#endif

	printf("Initialization took %.3f sec.\nMultiplication took %.3f sec.\nSaving took %.3f sec.\n",
			init_time / 1000000.0f, mult_time / 1000000.0f, save_time / 1000000.0f);
}


void ooc_square(	const ulong * primes, const uint total_primes,
		const ulong limit, const uint files, const uint bundle,
		const char * resultname, const char * name1, const char flags)
{
	const ulong blocksize = limit / files;
	const ulong length = limit / bundle;
	const ulong bundlesize = blocksize / bundle;
	const ulong size = sizeof(ulong) * max_threads * bundlesize;
	const uint num_prime_blocks = total_primes / max_threads;

	struct timeval begin, end;
	ulong exec_time, init_time = 0, sqr_time = 0, save_time = 0;

	nmod_poly_t fp[max_threads];

	#pragma omp parallel
	{
		const int n = omp_get_thread_num();
		nmod_poly_init(fp[n], 1);
		nmod_poly_fit_length(fp[n], length << 1);
	}

	for (uint i = 0; i < num_prime_blocks; i++)
	{
		#ifdef DEBUG
		printf("Block %u of %u\n", i, num_prime_blocks - 1);
		#endif

		gettimeofday(&begin, NULL);
		#pragma omp parallel for
		for (int s = 0; s < files; s++)
		{
			char name[150];

			sprintf(name, "%s%d", name1, s);
			int fd = open(name, O_RDONLY);

			if (fd == -1)
			{
					perror("In ooc_square: unable to open a file for reading.\n");
					exit(1);
			}

			ulong * file = (ulong *) mmap(0, sizeof(ulong) * max_threads * bundlesize, PROT_READ, MAP_SHARED, fd, sizeof(ulong) * bundlesize * (num_prime_blocks - 1 - i) * max_threads);

			if (file == MAP_FAILED)
			{
				close(fd);
				printf("In ooc_square: mapping failed when reading.\n");
			}

			for (int n = 0; n < max_threads; n++)
			{
				memcpy(fp[max_threads - 1 - n]->coeffs + s * bundlesize, file + n * bundlesize, sizeof(ulong) * bundlesize);
			}

			munmap(file, sizeof(ulong) * max_threads * bundlesize);

			#ifndef KEEEP_FILES
			ftruncate(fd, (num_prime_blocks - 1 - i) * size);
			#endif

			close(fd);
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		init_time += exec_time;
		#ifdef DEBUG
		printf("Init: %.3f\t%.3f\n", exec_time / 1000000.0, init_time / 1000000.0);
		#endif

		gettimeofday(&begin, NULL);
		#pragma omp parallel
		{
			const int n = omp_get_thread_num();

			ulong prime = primes[i * max_threads + n];
			nmod_t mod;

			mod.n = prime;
			mod.ninv = n_preinvert_limb(prime);
			count_leading_zeros(mod.norm, prime);

			fp[n]->length = length;

			fp[n]->mod = mod;

			if (flags & NO_TRUNCATE)
			{
				nmod_poly_pow(fp[n], fp[n], 2);
				fp[n]->length = (length << 1) - 1;
				fp[n]->coeffs[(length << 1) - 1] = 0;
			}
			else
			{
				nmod_poly_pow_trunc(fp[n], fp[n], 2, length);
			}

			flint_cleanup();
		}
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		sqr_time += exec_time;
		#ifdef DEBUG
		printf("Sqr:  %.3f\t%.3f\n", exec_time / 1000000.0, sqr_time / 1000000.0);
		#endif

		if ((flags & NO_TRUNCATE) == 0)
		{
			gettimeofday(&begin, NULL);
			#pragma omp parallel for
			for (int s = 0; s < files; s++)
			{
				char name[150];

				sprintf(name, "%s%u", resultname, s);
				int fd = open(name, O_RDWR | O_CREAT | O_APPEND, 0644);

				if (fd == -1)
				{
					perror("In ooc_square: unable to open a file for writing.\n");
					exit(1);
				}

				ftruncate(fd, (i + 1) * size);

				ulong * file = (ulong *) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, i * size);

				if (file == MAP_FAILED)
				{
					close(fd);
					perror("In ooc_square: mapping failed when writing.\n");
					exit(1);
				}

				for (int n = 0; n < max_threads; n++)
				{
					memcpy(file + n * bundlesize, fp[n]->coeffs + s * bundlesize, sizeof(ulong) * bundlesize);
				}

				munmap(file, size);

				close(fd);
			}
			gettimeofday(&end, NULL);
			exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
			save_time += exec_time;
			#ifdef DEBUG
			printf("Save: %.3f\t%.3f\n\n", exec_time / 1000000.0, save_time / 1000000.0);
			#endif
		}
		else
		{
			gettimeofday(&begin, NULL);
			#pragma omp parallel for
			for (int s = 0; s < files; s++)
			{
				char name[150];

				sprintf(name, "%s%u", resultname, s);
				int fd = open(name, O_RDWR | O_CREAT | O_APPEND, 0644);

				if (fd == -1)
				{
					perror("In ooc_square: unable to open a file for writing.\n");
					exit(1);
				}

				ftruncate(fd, (i + 1) * size << 1);

				ulong * file = (ulong *) mmap(0, size << 1, PROT_READ | PROT_WRITE, MAP_SHARED, fd, i * size << 1);

				if (file == MAP_FAILED)
				{
					close(fd);
					perror("In ooc_square: mapping failed when writing.\n");
					exit(1);
				}

				for (int n = 0; n < max_threads; n++)
				{
					memcpy(file + n * (bundlesize << 1), fp[n]->coeffs + s * (bundlesize << 1), sizeof(ulong) * bundlesize << 1);
				}

				munmap(file, size << 1);

				close(fd);
			}
			gettimeofday(&end, NULL);
			exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
			save_time += exec_time;
			#ifdef DEBUG
			printf("Save: %.3f\t%.3f\n\n", exec_time / 1000000.0, save_time / 1000000.0);
			#endif
		}
	}

	for (int n = 0; n < max_threads; n++)
	{
		nmod_poly_clear(fp[n]);
	}

	#ifndef KEEP_FILES
	for (int s = 0; s < files; s++)
	{
		char name[150];
		sprintf(name, "%s%d", name1, s);
		remove(name);
	}
	#endif 

	printf("Initialization took %.3f sec.\nSquaring took %.3f sec.\nSaving took %.3f sec.\n",
			init_time / 1000000.0f, sqr_time / 1000000.0f, save_time / 1000000.0f);
}


void restore_coeff(	const ulong * primes, const uint total_primes,
			const ulong limit, const uint files, const uint bundle, const uint bitsize, const int mod,
			const char * resultname, const char * name1, const int flags)
{
	// constants

	const ulong blocksize = limit / files;
	const ulong bundlesize = blocksize / bundle;
	const ulong threadsize = (bundlesize / max_threads) * bundle;
	const uint modulus = abs(mod);
	const int half = modulus >> 1;


	// variables for statistics

	struct timeval begin, end;

	ulong exec_time, crt_time = 0, rest_time = 0, save_time = 0;

	gettimeofday(&begin, NULL);
	char name[300];

	const ulong D = bitsize * bundle;

	// if the size of resulting coefficients exceeds 32 bits, save longs.
	// otherwise save ints.
	const size_t res_size = (bitsize > 32) ? sizeof(long) : sizeof(int);
	const size_t step = threadsize * (res_size / sizeof(int));

	fmpz_t bitmask;
	fmpz_init(bitmask);
	fmpz_one(bitmask);
	fmpz_mul_2exp(bitmask, bitmask, bitsize);
	fmpz_sub_ui(bitmask, bitmask, 1);

	fmpz_t blockmask;
	fmpz_init(blockmask);
	fmpz_one(blockmask);
	fmpz_mul_2exp(blockmask, blockmask, D);
	fmpz_sub_ui(blockmask, blockmask, 1);


	// Allocating space for each thread

	//fmpz_comb_t comb;
	//fmpz_comb_init(comb, primes, total_primes);

	fmpz_comb_temp_t comb_temp[max_threads];

	ulong * thread_residues[max_threads];

	int * result[max_threads];

	fmpz * a_threads[max_threads];
	fmpz * b_threads[max_threads];

	fmpz * d = _fmpz_vec_init(bundlesize);

	#pragma omp parallel
	{
		const int n = omp_get_thread_num();

		fmpz_comb_temp_init(comb_temp[n], comb);

		thread_residues[n] = (ulong *) malloc(sizeof(ulong) * total_primes);

		if (thread_residues[n] == NULL)
		{
			perror("In restore_coeff: unable to allocate thread_residues.\n");
			exit(1);
		}

		result[n] = (int *) malloc(res_size * (threadsize + ((n == max_threads - 1) ? (blocksize - max_threads * threadsize) : 0)));

		a_threads[n] = _fmpz_vec_init(2);
		_fmpz_vec_zero(a_threads[n], 2);
		b_threads[n] = _fmpz_vec_init(2);
		_fmpz_vec_zero(b_threads[n], 2);
	}

	fmpz * temp = _fmpz_vec_init(2);
	_fmpz_vec_zero(temp, 2);

	ulong * file;

	ulong * addr[total_primes];

	// file counter
	int s = 0;

	if (flags & WITH_INVERSE)
	{
		char filename[300];
		char command[300];

		for (int s = 0; s < files; s += 2)
		{
			sprintf(name, "%s%d", resultname, s);
			sprintf(filename, "%s%d", resultname, s + 1);
			sprintf(command, "cat %s >> %s", filename, name);
			system(command);
			remove(filename);
			sprintf(filename, "%s%d", resultname, s >> 1);
			rename(name, filename);
		}

		s = (files >> 1);
	}

	if (flags & WITH_SQUARING)
	{
		char filename[300];
		char command[300];

		sprintf(name, "%s%d", resultname, 0);

		if (access(name, F_OK) == 0)
		{
			for (int s = (files >> 1); s < files; s++)
			{
				sprintf(name, "%s%d", resultname, s);
				remove(name);
			}

			for (int s = 0; s < (files >> 1); s += 2)
			{
				sprintf(name, "%s%d", resultname, s);
				sprintf(filename, "%s%d", resultname, s + 1);
				sprintf(command, "cat %s >> %s", filename, name);
				system(command);
				remove(filename);
				sprintf(filename, "%s%d", resultname, s >> 1);
				rename(name, filename);
			}

			s = (files >> 2);
		}
	}

	fmpz_t d_with_inverse;

	if (mod && ((s == files >> 1) || (s == files >> 2)))
	{
		fmpz_init(d_with_inverse);

		sprintf(name, "%s%d", name1, s - 1);
		int fd = open(name, O_RDONLY);

		if (fd == -1)
		{
			char str[200];
			sprintf(str, "In restore_coeff: unable to open a file %s for reading\n", name);
			perror(str);
			fflush(stderr);
			exit(1);
		}

		file = (ulong *) mmap(0, sizeof(ulong) * total_primes * bundlesize, PROT_READ, MAP_SHARED, fd, 0);

		if (file == MAP_FAILED)
		{
			close(fd);
			perror("In restore_coeff: mapping to read a file failed\n");
			exit(1);
		}

		ulong * residues = thread_residues[0];

		for (uint i = 0; i < total_primes; i++)
		{
			residues[i] = file[(i + 1) * bundlesize - 1];
		}

		fmpz_multi_CRT_ui(d_with_inverse, residues, comb, comb_temp[0], mod != 0);

		close(fd);

		unsigned char carry_prev = 0, carry_cur = 0;

		for (uint i = 0; i < 2; i++)
		{
			fmpz_and(temp + i, d_with_inverse, blockmask);

			if (mod && fmpz_tstbit(temp + i, D - 1))
			{
				carry_cur = 1;
				fmpz_complement(temp + i, temp + i);
				fmpz_and(temp + i, temp + i, blockmask);
				fmpz_add_ui(temp + i, temp + i, 1);
				fmpz_neg(temp + i, temp + i);
			}

			if (carry_prev)
			{
				fmpz_add_ui(temp + i, temp + i, 1);
			}

			carry_prev = carry_cur;
			carry_cur = 0;

			if (i != 1)
			{
				fmpz_fdiv_q_2exp(d_with_inverse, d_with_inverse, D);
			}
		}

		fmpz_clear(d_with_inverse);

		#ifndef KEEP_FILES
		char filename[150];

		for (int t = 0; t < s; t++)
		{
			sprintf(filename, "%s%d", name1, t);
			remove(filename);
		}
		#endif
	}
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Initialization took %.3f sec.\n", exec_time / 1000000.0f);

	for (; s < files; s++)
	{
		#ifdef DEBUG
		printf("File %d of %d\n", s, files - 1);
		#endif

		// Opening next file to read from
		gettimeofday(&begin, NULL);
		sprintf(name, "%s%d", name1, s);
		int fd = open(name, O_RDONLY);

		if (fd == -1)
		{
			perror("In restore_coeff: unable to open a file for reading.\n");
			exit(1);
		}

		file = (ulong *) mmap(0, sizeof(ulong) * total_primes * bundlesize, PROT_READ, MAP_SHARED, fd, 0);

		if (file == MAP_FAILED)
		{
			close(fd);
			perror("In restore_coeff: mapping to read a file failed\n");
			exit(1);
		}

		for (uint i = 0; i < total_primes; i++)
		{
			addr[i] = file + i * bundlesize;
		}

		#pragma omp parallel
		{
			const int n = omp_get_thread_num();

			ulong * residues = thread_residues[n];

			#pragma omp for
			for (uint j = 0; j < bundlesize; j++)
			{
				for (uint i = 0; i < total_primes; i++)
				{
					residues[i] = addr[i][j];
				}

				fmpz_multi_CRT_ui(&d[j], residues, comb, comb_temp[n], 1);
			}
		}

		munmap(file, sizeof(ulong) * total_primes * bundlesize);
		close(fd);

		sprintf(name, "%s%d", name1, s);
		#ifndef KEEP_FILES
		remove(name);
		#endif
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		crt_time += exec_time;
		#ifdef DEBUG
		printf("CRT:  %.3f\t%.3f\n", exec_time / 1000000.0f, crt_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);

		#pragma omp parallel
		{
			const int n = omp_get_thread_num();

			long coeff = 0;

			int * res = result[n];

			size_t size = (bundlesize / max_threads) + ((n == max_threads - 1) ? (bundlesize % max_threads) : 0);

			unsigned char carry_prev, carry_cur;


			// variables for restoration of coefficients

			int cur_entry = 0;

			fmpz * a = a_threads[n];
			fmpz * b = b_threads[n];

			fmpz_t t;
			fmpz_init(t);
			fmpz_t T;
			fmpz_init(T);
			fmpz_t reconst_val;
			fmpz_init(reconst_val);

			ulong cur = n * threadsize + s * blocksize;

			for (uint k = 0; k < size; k++)
			{
				if (n > 0 || k > 0)
				{
					if (k == 0)
					{
						fmpz_set(t, &d[n * (bundlesize / max_threads) - 1]);

						fmpz_and(b, t, blockmask);
						if (mod && fmpz_tstbit(b, D - 1))
						{
							fmpz_complement(b, b);
							fmpz_and(b, b, blockmask);
							fmpz_add_ui(b, b, 1);
							fmpz_neg(b, b);
						}

						fmpz_fdiv_q_2exp(t, t, D);
						fmpz_and(b + 1, t, blockmask);
						if (mod && fmpz_tstbit(b + 1, D - 1))
						{
							fmpz_complement(b + 1, b + 1);
							fmpz_and(b + 1, b + 1, blockmask);
							fmpz_add_ui(b + 1, b + 1, 1);
							fmpz_neg(b + 1, b + 1);
						}

						if (mod && fmpz_tstbit(b, D - 1))
						{
							fmpz_add_ui(b + 1, b + 1, 1);
						}

						// DEBUG
						fmpz_fdiv_q_2exp(t, t, D);
						if (fmpz_cmp_ui(t, 0) > 0)
						{
							printf("1 NOTE: %ld: ", D);
							fmpz_print(t);
							printf("\n");
							fflush(stdout);
							exit(1);
						}
					}
				}
				else
				{
					_fmpz_vec_set(b, temp, 2);
				}

				fmpz_set(t, &d[n * (bundlesize / max_threads) + k]);

				fmpz_and(a, t, blockmask);
				if (mod && fmpz_tstbit(a, D - 1))
				{
					fmpz_complement(a, a);
					fmpz_and(a, a, blockmask);
					fmpz_add_ui(a, a, 1);
					fmpz_neg(a, a);
				}

				fmpz_fdiv_q_2exp(t, t, D);
				fmpz_and(a + 1, t, blockmask);
				if (mod && fmpz_tstbit(a + 1, D - 1))
				{
					fmpz_complement(a + 1, a + 1);
					fmpz_and(a + 1, a + 1, blockmask);
					fmpz_add_ui(a + 1, a + 1, 1);
					fmpz_neg(a + 1, a + 1);
				}

				if (mod && fmpz_tstbit(a, D - 1))
				{
					fmpz_add_ui(a + 1, a + 1, 1);
				}

				fmpz_fdiv_q_2exp(t, t, D);
				if (fmpz_cmp_ui(t, 1) > 0)
				{
					printf("2 NOTE: %ld: ", D);
					fmpz_print(t);
					printf("\n");
					fflush(stdout);
					exit(1);
				}

				fmpz_add(T, a, b + 1);

				for (uint i = 0; i < bundle; i++)
				{
					fmpz_and(reconst_val, T, bitmask);
					fmpz_fdiv_q_2exp(T, T, bitsize);

					//printf("%lu: %lu (%lu)\n", cur, fmpz_get_ui(reconst_val), cur_disc);

					if (mod)
					{
						if (fmpz_tstbit(reconst_val, bitsize - 1))
						{
							fmpz_complement(reconst_val, reconst_val);
							fmpz_and(reconst_val, reconst_val, bitmask);
							fmpz_add_ui(reconst_val, reconst_val, 1);
							fmpz_neg(reconst_val, reconst_val);
							carry_cur = 1;
						}

						if (carry_prev)
						{
							fmpz_add_ui(reconst_val, reconst_val, 1);
						}

						carry_prev = carry_cur;
						carry_cur = 0;

						//printf("%lu: ", cur);
						//fmpz_print(reconst_val);

						coeff = fmpz_fdiv_ui(reconst_val, modulus);

						if ((flags & WITH_INVERSE) && cur >= (limit >> 1))
						{
							coeff = -coeff;

							coeff += ((-coeff) / modulus + 1) * modulus;

							if (coeff >= modulus)
							{
								coeff -= modulus;
							}
						}

						if (mod < 0 && coeff > half)
						{
							coeff -= modulus;
						}

						//printf(" (%ld)\n", coeff);
					}
					else
					{
						coeff = fmpz_get_ui(reconst_val);
					}

					//printf("%lu %ld\n", cur, coeff);

					if (bitsize > 32)
					{
						res[cur_entry++] = (int) (coeff & 0xFFFFFFFF);
						res[cur_entry++] = (int) (coeff >> 32);
					}
					else
					{
						res[cur_entry++] = (int) coeff;
					}

					cur++;
				}

				carry_prev = 0;

				fmpz_set(b, a);
				fmpz_set(b + 1, a + 1);
			}

			fmpz_clear(reconst_val);
			fmpz_clear(t);
			fmpz_clear(T);
		}

		_fmpz_vec_set(temp, a_threads[max_threads - 1], 2);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		rest_time += exec_time;
		#ifdef DEBUG
		printf("Rest: %.3f\t%.3f\n", exec_time / 1000000.0f, rest_time / 1000000.0f);
		#endif

		gettimeofday(&begin, NULL);

		sprintf(name, "%s%d", resultname, s);
		fd = open(name, O_RDWR | O_CREAT, 0644);

		if (fd == -1)
		{
			perror("In restore_coeff: unable to open a file for writing.\n");
			exit(1);
		}

		size_t size = res_size * blocksize;

		ftruncate(fd, size);

		int * file = (int *) mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

		if (file == MAP_FAILED)
		{
			close(fd);
			perror("In restore_coeff: mapping failed when writing.\n");
			exit(1);
		}

		off_t thread_offset = 0;

		for (int n = 0; n < max_threads; n++)
		{
			memcpy(file + thread_offset, result[n], res_size * (threadsize + ((n == max_threads - 1) ? (blocksize - max_threads * threadsize) : 0)));
			thread_offset += step;
		}

		munmap(file, size);

		close(fd);

		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		save_time += exec_time;
		#ifdef DEBUG
		printf("Save: %.3f\t%.3f\n", exec_time / 1000000.0f, save_time / 1000000.0f);
		#endif
	}

	gettimeofday(&begin, NULL);
	#pragma omp parallel
	{
		const int n = omp_get_thread_num();
		free(result[n]);
		free(thread_residues[n]);
		fmpz_comb_temp_clear(comb_temp[n]);
		_fmpz_vec_clear(a_threads[n], 2);
		_fmpz_vec_clear(b_threads[n], 2);
	}

	_fmpz_vec_clear(d, bundlesize);

	_fmpz_vec_clear(temp, 2);

	//fmpz_comb_clear(comb);
	fmpz_clear(bitmask);
	fmpz_clear(blockmask);

	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);

	printf("CRT took %.3f sec.\n", crt_time / 1000000.0f);

	printf("Coeff recovery took %.3f sec.\nSaving took %.3f sec.\nClearing took %.3f sec.\n",
			rest_time / 1000000.0f, save_time / 1000000.0f, exec_time / 1000000.0f);
}

void invert(	const ulong * primes,
		const uint maxpow, const uint files, const uint bundle, const int mod,
		const char * resultname, const char * folder, const char type)
{
	uint new_bundle = 0, new_bitsize = 0, output_bits = 0, total_primes = 0;

	struct timeval begin, end;
	ulong exec_time, sq_time = 0, mult_time = 0, rest1_time = 0, rest2_time = 0;

	ulong pow = (maxpow > MINPOW) ? MINPOW : maxpow;

	ulong limit = 1L << pow;

	gettimeofday(&begin, NULL);
	nmod_poly_t poly;

	printf("Initializing the polynomial %u to %lu...\n", type, limit);

	nmod_poly_init(poly, mod);

	if (type == THETA3)
	{
		nmod_poly_theta3(poly, limit);
	}
	else if (type == THETA234)
	{
		nmod_poly_theta234(poly, limit);
	}
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Initialized in %.3f sec.\n", exec_time / 1000000.0f);

	gettimeofday(&begin, NULL);
	printf("Inverting the polynomial to %lu modulo %u using flint...\n", limit, mod);
	nmod_poly_inv_series(poly, poly, limit);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Inverted in %.3f sec.\n", exec_time / 1000000.0f);

	gettimeofday(&begin, NULL);
	printf("Saving to files...\n");
	nmod_poly_to_files(poly, limit, files, resultname);
	nmod_poly_clear(poly);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Saved in %.3f sec.\n\n", exec_time / 1000000.0f);
	gettimeofday(&end, NULL);


	char name_g[160];
	sprintf(name_g, "%s/g", folder);
	mkdir(name_g, 0777);
	sprintf(name_g, "%s%s", folder, "/g/g");

	char name_h[160];
	sprintf(name_h, "%s/h", folder);
	mkdir(name_h, 0777);
	sprintf(name_h, "%s%s", folder, "/h/h");

	char name_i[160];
	sprintf(name_i, "%s/i", folder);
	mkdir(name_i, 0777);
	sprintf(name_i, "%s%s", folder, "/i/i");

	char name_tmp[160];
	sprintf(name_tmp, "%s/j", folder);
	mkdir(name_tmp, 0777);
	sprintf(name_tmp, "%s%s", folder, "/j/j");

	if (maxpow > MINPOW)
	{
		printf("Launching out-of-core inversion...\n");
	}

	for (uint i = MINPOW + 1; i <= maxpow; i++)
	{
		printf("Inversion %u of %u\n", i, maxpow);

		limit = 1L << i;

		new_bundle = MAX(bundle >> (maxpow - i), 2);
		new_bitsize =  2 * ((uint) ceil(log2(mod))) + i - 1;
		output_bits = new_bitsize * ((new_bundle << 1) - 1);
		total_primes = (uint) ceil((output_bits - 1) / log2(primes[0]));
		total_primes += (max_threads - total_primes % max_threads);

		printf("Limit: %lu\nBundle: %u\nBitsize: %u\nTotal primes: %u\n", limit, new_bundle, new_bitsize, total_primes);

		init_files(primes, total_primes, limit >> 1, files, new_bundle, new_bitsize, mod, name_g, resultname, NO_REMOVE);

		gettimeofday(&begin, NULL);
		printf("Squaring...\n");
		ooc_square(primes, total_primes, limit >> 1, files, new_bundle, name_tmp, name_g, NO_TRUNCATE);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		sq_time += exec_time;
		printf("Squared in %.3f sec.\n\n", exec_time / 1000000.0);

		gettimeofday(&begin, NULL);
		printf("Restoring...\n");
		restore_coeff(primes, total_primes, limit, files, new_bundle, new_bitsize, mod, name_i, name_tmp, WITH_SQUARING);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		rest1_time += exec_time;
		printf("Restored in %.3f sec.\n\n", exec_time / 1000000.0);

		init_files(primes, total_primes, limit, files, new_bundle, new_bitsize, mod, name_g, name_i, (i != maxpow) ? NO_REMOVE : 0);

		init_files(primes, total_primes, limit, files, new_bundle, new_bitsize, mod, name_h, (void *) ((ulong) type), 0);

		gettimeofday(&begin, NULL);
		printf("Multiplying...\n");
		ooc_multiply(primes, total_primes, limit, files, new_bundle, name_tmp, name_g, name_h);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		mult_time += exec_time;
		printf("Multiplied in %.3f sec.\n\n", exec_time / 1000000.0);

		gettimeofday(&begin, NULL);
		printf("Restoring...\n");
		restore_coeff(primes, total_primes, limit, files, new_bundle, new_bitsize, -mod, resultname, name_tmp, WITH_INVERSE);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		rest2_time += exec_time;
		printf("Restored in %.3f sec.\n\n", exec_time / 1000000.0);
	}
}

void multiply(const ulong limit, const uint files, const uint bundle, const ulong bound,
		const char * resultname, const char * folder, const char type1, const char type2)
{
	ulong exec_time;

	struct timeval begin, end;

	max_threads = omp_get_max_threads();

	printf("\nTotal threads: %d\n", max_threads);

	ulong p0 = n_nextprime(1L << 62, 1);

	int mod = 0;

	uint bitsize = (log2(bound) + 1);
	uint output_bits = bitsize * ((bundle << 1) - 1);

	uint total_primes = (uint) ceil((output_bits - 1) / log2(p0));
	total_primes += (max_threads - total_primes % max_threads);

	ulong * primes = malloc(sizeof(ulong) * total_primes);

	primes[0] = p0;

	init_primes(primes + 1, total_primes - 1, p0 + 1);

	fmpz_comb_init(comb, primes, total_primes);

	printf("Limit: %lu\nFiles: %u\nBundle: %u\nBound: %lu\nBitsize: %u\nTotal primes: %u\n", limit, files, bundle, bound, bitsize, total_primes);

	char name_f[160];
	sprintf(name_f, "%s/%s", folder, "f");
	mkdir(name_f, 0777);
	sprintf(name_f, "%s/%s", name_f, "f");

	char name_g[160];
	if (type1 != type2)
	{
		sprintf(name_g, "%s/%s", folder, "g");
		mkdir(name_g, 0777);
		sprintf(name_g, "%s/%s", name_g, "g");
	}

	char name_h[160];
	sprintf(name_h, "%s/%s", folder, "h");
	mkdir(name_h, 0777);
	sprintf(name_h, "%s/%s", name_h, "h");

	char result[160];
	sprintf(result, "%s/%s", folder, resultname);

	if (type1 != type2)
	{
		gettimeofday(&begin, NULL);
		printf("Initializing %u...\n", type1);
		init_files(primes, total_primes, limit, files, bundle, bitsize, mod, name_g, (const void *) (ulong) type1, 0);
		gettimeofday(&end, NULL);
		exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
		printf("Initialized in %.3f sec.\n\n", exec_time / 1000000.0);
	}

	gettimeofday(&begin, NULL);
	printf("Initializing %u...\n", type2);
	init_files(primes, total_primes, limit, files, bundle, bitsize, mod, name_f, (const void *) (ulong) type2, 0);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Initialized in %.3f sec.\n\n", exec_time / 1000000.0);

	gettimeofday(&begin, NULL);
	if (type1 != type2)
	{
		printf("Multiplying...\n");
		ooc_multiply(primes, total_primes, limit, files, bundle, name_h, name_f, name_g);
	}
	else
	{
		printf("Squaring...\n");
		ooc_square(primes, total_primes, limit, files, bundle, name_h, name_f, 0);
	}
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	if (type1 != type2)
	{
		printf("Multiplied in %.3f sec.\n\n", exec_time / 1000000.0);
	}
	else
	{
		printf("Squared in %.3f sec.\n\n", exec_time / 1000000.0);
	}

	gettimeofday(&begin, NULL);
	printf("Restoring...\n");
	restore_coeff(primes, total_primes, limit, files, bundle, bitsize, mod, result, name_h, 0);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Restored in %.3f sec.\n\n", exec_time / 1000000.0);

	free(primes);

	fmpz_comb_clear(comb);
}



void divide(const ulong limit, const uint files, const uint bundle, const ulong bound,
		const char * resultname, const char * folder, const char type1, const char type2)
{
	ulong exec_time;

	struct timeval begin, end;

	max_threads = omp_get_max_threads();

	printf("\nTotal threads: %d\n", max_threads);

	ulong p0 = n_nextprime(1L << 62, 1);

	int mod = n_nextprime(bound & -2, 1);

	uint bitsize = 2 * ((uint) ceil(log2(mod))) + log2(limit) - 1;
	uint output_bits = bitsize * ((bundle << 1) - 1);

	uint total_primes = (uint) ceil((output_bits - 1) / log2(p0));
	total_primes += (max_threads - total_primes % max_threads);

	ulong * primes = malloc(sizeof(ulong) * total_primes);

	primes[0] = p0;

	init_primes(primes + 1, total_primes - 1, p0 + 1);

	fmpz_comb_init(comb, primes, total_primes);

	printf("Limit: %lu\nFiles: %u\nBundle: %u\nBound: %lu\nBitsize: %u\nTotal primes: %u\n", limit, files, bundle, bound, bitsize, total_primes);

	char name_f[160];
	sprintf(name_f, "%s/%s", folder, "f/f");

	char name_g[160];
	sprintf(name_g, "%s/%s", folder, "g/g");

	char name_h[160];
	sprintf(name_h, "%s/%s", folder, "h/h");

	char result[160];
	sprintf(result, "%s/%s", folder, resultname);

	gettimeofday(&begin, NULL);
	printf("Inverting %u...\n", type2);
	invert(primes, log2(limit), files, bundle, mod, name_f, folder, type2);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Inverted in %.3f sec.\n\n", exec_time / 1000000.0);

	gettimeofday(&begin, NULL);
	printf("Initializing from %s...\n", name_f);
	init_files(primes, total_primes, limit, files, bundle, bitsize, mod, name_f, name_f, 0);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Initialized in %.3f sec.\n\n", exec_time / 1000000.0);

	gettimeofday(&begin, NULL);
	printf("Initializing %u...\n", type1);
	init_files(primes, total_primes, limit, files, bundle, bitsize, mod, name_g, (const void *) (ulong) type1, 0);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Initialized in %.3f sec.\n\n", exec_time / 1000000.0);

	gettimeofday(&begin, NULL);
	printf("Multiplying...\n");
	ooc_multiply(primes, total_primes, limit, files, bundle, name_h, name_f, name_g);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Multiplied in %.3f sec.\n\n", exec_time / 1000000.0);

	gettimeofday(&begin, NULL);
	printf("Restoring...\n");
	restore_coeff(primes, total_primes, limit, files, bundle, bitsize, mod, result, name_h, 0);
	gettimeofday(&end, NULL);
	exec_time = (end.tv_sec*1e6 + end.tv_usec) - (begin.tv_sec*1e6 + begin.tv_usec);
	printf("Restored in %.3f sec.\n\n", exec_time / 1000000.0);

	free(primes);

	fmpz_comb_clear(comb);
}
