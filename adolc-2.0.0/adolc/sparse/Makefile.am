##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 273 2008-12-15 14:44:29Z awalther $
##
## Copyright (C) 2008
##               Technical University Dresden
##               Department of Mathematics
##               Institute of Scientific Computing
##
## This file is part of ADOL-C. This software is provided as open source.
## Any use, reproduction, or distribution of the software constitutes 
## recipient's acceptance of the terms of the accompanying license file.
## 
##############################################################################

AM_CFLAGS                = @ac_adolc_cflags@ -DADOLC_INTERNAL=1
AM_CXXFLAGS              = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1
AM_LDFLAGS               = -no-undefined

INCLUDES                 = -I$(top_builddir)

MAINTAINERCLEANFILES     = Makefile.in *~ *.orig

EXTRA_DIST               = Makefile.win

noinst_LTLIBRARIES       = libsparse.la

libsparseincludedir      = $(pkgincludedir)/sparse

libsparseinclude_HEADERS = sparsedrivers.h sparse_fo_rev.h 

libsparse_la_SOURCES     = sparse_fo_rev.cpp sparsedrivers.cpp
