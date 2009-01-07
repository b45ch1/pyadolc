##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 270 2008-12-15 13:04:25Z awalther $
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

AM_CFLAGS            = @ac_adolc_cflags@
AM_CXXFLAGS          = @ac_adolc_cxxflags@

if SPARSE
LDADD                = -ladolc $(libdir)/libColPack.a
AM_LDFLAGS           = -L../../../adolc
else
LDADD                = -ladolc 
AM_LDFLAGS           = -L../../../adolc
endif

INCLUDES             = -I$(top_builddir)

MAINTAINERCLEANFILES = Makefile.in *~ *.orig

EXTRA_DIST           = Makefile.win

if ADDEXA
noinst_PROGRAMS      = powexam

powexam_SOURCES      = powexam.cpp
endif
