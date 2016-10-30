AUTOMAKE_OPTIONS = foreign
CFLAGS=-Wall -fopenmp -std=gnu99 -O3
LDFLAGS=
bin_PROGRAMS = polymult
lib_LIBRARIES = libpolymult.a
libpolymult_a_SOURCES = init.c mult.c
polymult_SOURCES = init.c mult.c main.c
polymultincludedir = $(includedir)/libpolymult
polymultinclude_HEADERS = init.h mult.h
