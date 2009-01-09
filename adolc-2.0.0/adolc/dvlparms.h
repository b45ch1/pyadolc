/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     dvlparms.h
 Revision: $Id: dvlparms.h 278 2008-12-19 09:11:26Z awalther $
 Contents: Developer parameters:
           These parameters are intended for use by the developers and 
           maintainers of ADOL-C to specify library wide definitions.
 
 Copyright (c) 2008
               Technical University Dresden
               Department of Mathematics
               Institute of Scientific Computing
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#if !defined(ADOLC_DVLPARMS_H)
#define ADOLC_DVLPARMS_H 1

/*--------------------------------------------------------------------------*/
/* File names for the tapes */
#define ADOLC_TAYLORS_NAME    "ADOLC-Taylors_"
#define ADOLC_VALUES_NAME     "ADOLC-Values_"
#define ADOLC_LOCATIONS_NAME  "ADOLC-Locations_"
#define ADOLC_OPERATIONS_NAME "ADOLC-Operations_"

/****************************************************************************/
/* Tells ADOL-C which name space offers the mathematical functions          */
#define ADOLC_MATH_NSP std

/*--------------------------------------------------------------------------*/
/* TAPE IDENTIFICATION (ADOLC & version check) */
#define statSpace   22
#define adolcIDSize  5
/* NOTE: adolcIDSize + statSize <= statSpace required! */

/*--------------------------------------------------------------------------*/
/* ADOL-C configuration (never change this) */
#define overwrite 1
#define compsize >

/*--------------------------------------------------------------------------*/
#endif
