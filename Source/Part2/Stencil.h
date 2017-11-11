/***
 * File: Sobel.h
 * Desc: Data type for top optimize extracting a 3x3 stencil from a larger double array
 */
 #ifndef STENCIL_H
 #define STENCIL_H

typedef struct
{
   double *top;
   double *middle;
   double *bottom;
} Stencil_t;

 #endif
