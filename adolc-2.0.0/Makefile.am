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

AUTOMAKE_OPTIONS     = gnu

EXTRA_DIST           = TODO BUGS Makefile.win make_32 make_64 \
                       comp_win.bat winflags_dll winflags_exa

if DOCEXA
SUBDIRS              = adolc examples doc
else
if ADDEXA
SUBDIRS              = adolc examples doc
else
if PAREXA
SUBDIRS              = adolc examples doc
else
SUBDIRS              = adolc doc
endif
endif
endif

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 configure adolc/config-h.in \
                       adolc/stamp-h.in *~ *.orig
