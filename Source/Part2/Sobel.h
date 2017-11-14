/***
 * File: Sobel.h
 * Desc: Functions to apply a Sobel operator to a stencil.
 */

#ifndef SOBEL_H
#define SOBEL_H

#include "Stencil.h"

/*
 * Convolution grid to calculate the Sobel edge gradient in the X-direction
 */
extern const int Sobel_Gx[3][3];

/*
* Convolution grid to calculate the Sobel edge gradient in the Y-direction
*/
extern const int Sobel_Gy[3][3];

/*
 * Calculates the magnitude of the gradient of the stencil of image values
 * @param stencil -- the 3x3 image stencil
 * @return -- magnitude of the gradient
 */
double Sobel_Magnitude(Stencil_t *stencil);

#endif
