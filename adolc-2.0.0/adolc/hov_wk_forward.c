/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     hov_wk_forward.c
 Revision: $Id: hov_wk_forward.c 278 2008-12-19 09:11:26Z awalther $
 Contents: hov_wk_forward (higher-order-vector forward mode with keep)
 
 Copyright (c) 2008
               Technical University Dresden
               Department of Mathematics
               Institute of Scientific Computing
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

 ----------------------------------------------------------------------------*/
#define _HOV_WK_ 1
#define _KEEP_   1
#include <adolc/uni5_for.c>
#undef _KEEP_
#undef _HOV_WK_
