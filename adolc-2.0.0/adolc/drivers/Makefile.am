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

AM_CFLAGS                 = @ac_adolc_cflags@ -DADOLC_INTERNAL=1
AM_CXXFLAGS               = @ac_adolc_cxxflags@ -DADOLC_INTERNAL=1
AM_LDFLAGS                = -no-undefined

MAINTAINERCLEANFILES      = Makefile.in *~ *.orig

EXTRA_DIST                = Makefile.win

INCLUDES                  = -I$(top_builddir)

noinst_LTLIBRARIES        = libdrivers.la

libdriversincludedir      = $(pkgincludedir)/drivers

libdriversinclude_HEADERS = drivers.h odedrivers.h taylor.h

libdrivers_la_SOURCES     = drivers.c driversf.c odedrivers.c odedriversf.c \
                            taylor.c
