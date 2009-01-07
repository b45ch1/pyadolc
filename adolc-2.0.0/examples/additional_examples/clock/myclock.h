/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     myclock.h
 Revision: $Id: myclock.h 268 2008-12-15 10:32:03Z awalther $
 Contents: timing utilities

 Copyright (c) 2008
               Technical University Dresden
               Department of Mathematics
               Institute of Scientific Computing
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/
#ifndef _MYCLOCK_H_
#define _MYCLOCK_H_

/****************************************************************************/
/*                                                        CLOCKS PER SECOND */
extern double clocksPerSecond;


/****************************************************************************/
/*                                                                    CLOCK */
double myclock(int normalize = 0);


/****************************************************************************/
/*                                                          NORMALIZE CLOCK */
void normalizeMyclock( void );

#endif








