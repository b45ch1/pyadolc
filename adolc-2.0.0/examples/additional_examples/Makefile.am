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

MAINTAINERCLEANFILES = Makefile.in *~

EXTRA_DIST           = Makefile.win

if ADDEXA
SUBDIRS              = clock hessmat lufact ode sparse tapesave timing \
                       detexam helm lighthouse scal speelpenning taylor pow \
                       checkpointing ext_diff_func fixpoint_exam openmp_exam
endif

if PAREXA
SUBDIRS              = clock hessmat lufact ode sparse tapesave timing \
                       detexam helm lighthouse scal speelpenning taylor pow \
                       checkpointing ext_diff_func fixpoint_exam openmp_exam
endif
