/***
 * File: Sobel.h
 * Desc: Functions to apply a Sobel operator to a stencil.
 */
#ifndef SOBEL_H
#define SOBEL_H

/*
 * Calculates the magnitude of the gradient of the 3x3 stencil of image values
 */
double Sobel_Magnitude(double stencil[3][3]);

#endif
