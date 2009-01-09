/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     myclock.cpp
 Revision: $Id: myclock.cpp 268 2008-12-15 10:32:03Z awalther $
 Contents: timing utilities

 Copyright (c) 2008
               Technical University Dresden
               Department of Mathematics
               Institute of Scientific Computing
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
 
/****************************************************************************/
/*                                                                 INCLUDES */
#include <sys/timeb.h>
#include <time.h>
#include <clock/myclock.h>



/****************************************************************************/
/*                                                          CLOCK UTILITIES */

double myclock( int normalize ) {
    struct timeb tb;

    ftime(&tb);
    return ((double)tb.time+(double)tb.millitm/1000.);
}

void normalize() {}

