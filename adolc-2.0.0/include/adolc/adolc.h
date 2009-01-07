/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc.h
 Revision: $Id: adolc.h 278 2008-12-19 09:11:26Z awalther $
 Contents: Provides all C/C++ interfaces of ADOL-C.
           NOTICE: ALL C/C++ headers will be included DEPENDING ON 
           whether the source code is plain C or C/C++ code. 
 
 Copyright (c) 2008
               Technical University Dresden
               Department of Mathematics
               Institute of Scientific Computing

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_ADOLC_H)
#define ADOLC_ADOLC_H 1

#include <adolc/common.h>

/****************************************************************************/
/*                                                  Now the pure C++ THINGS */
#if defined(__cplusplus)
/*--------------------------------------------------------------------------*/
/* Operator overloading things (active doubles & vectors) */
#  include <adolc/adouble.h>
#  include <adolc/externfcts.h>
#  include <adolc/checkpointing.h>
#  include <adolc/fixpoint.h>
#endif

/****************************************************************************/
/*                                                     Now the C/C++ THINGS */

/*--------------------------------------------------------------------------*/
/* interfaces to basic forward/reverse routines */
#include <adolc/interfaces.h>

/*--------------------------------------------------------------------------*/
/* interfaces to "Easy To Use" driver routines for ... */
#include <adolc/drivers/drivers.h>    /* optimization & nonlinear equations */
#include <adolc/drivers/taylor.h>     /* higher order tensors &
inverse/implicit functions */
#include <adolc/drivers/odedrivers.h> /* ordinary differential equations */

/*--------------------------------------------------------------------------*/
/* interfaces to TAPEDOC package */
#include <adolc/tapedoc/tapedoc.h>

/*--------------------------------------------------------------------------*/
/* interfaces to SPARSE package */
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/sparse/sparse_fo_rev.h>

/*--------------------------------------------------------------------------*/
/* tape and value stack utilities */
#include <adolc/taping.h>

/*--------------------------------------------------------------------------*/
/* allocation utilities */
#include <adolc/adalloc.h>

/****************************************************************************/
#endif
