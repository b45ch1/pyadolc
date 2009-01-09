##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## Revision: $Id: Makefile.am 277 2008-12-19 08:02:52Z awalther $
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

AM_CFLAGS               = @ac_adolc_cflags@
AM_CXXFLAGS             = @ac_adolc_cxxflags@

if SPARSE
LDADD                   = -ladolc $(libdir)/libColPack.a
AM_LDFLAGS              = -L../adolc
else
LDADD                   = -ladolc 
AM_LDFLAGS              = -L../adolc
endif

INCLUDES                = -I$(top_builddir)

MAINTAINERCLEANFILES    = Makefile.in *~ *.orig ADOLC-*.tap

EXTRA_DIST              = Makefile.win

if ADDEXA
SUBDIRS                 = additional_examples
endif

if PAREXA
SUBDIRS                 = additional_examples
endif

if DOCEXA
noinst_PROGRAMS         = detexam speelpenning powexam odexam \
                          tapeless_scalar tapeless_vector

detexam_SOURCES         = detexam.cpp

speelpenning_SOURCES    = speelpenning.cpp

powexam_SOURCES         = powexam.cpp

odexam_SOURCES          = odexam.cpp 

tapeless_scalar_SOURCES = tapeless_scalar.cpp

tapeless_vector_SOURCES = tapeless_vector.cpp
endif
