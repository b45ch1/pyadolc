##############################################################################
## Makefile.am -- Process this file with automake to produce Makefile.in
## subdirectory tapedoc
## Revision: $Id: Makefile.am 272 2008-12-15 14:23:20Z awalther $
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

AM_CFLAGS                 = @ac_adolc_cflags@ -DADOLC_INTERNAL=1
AM_CXXFLAGS               = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1
AM_LDFLAGS                = -no-undefined

MAINTAINERCLEANFILES      = Makefile.in *~ *.orig

EXTRA_DIST                = Makefile.win

INCLUDES                  = -I$(top_builddir)

noinst_LTLIBRARIES        = libtapedoc.la

libtapedocincludedir      = $(pkgincludedir)/tapedoc

libtapedocinclude_HEADERS = tapedoc.h

libtapedoc_la_SOURCES     = tapedoc.c
