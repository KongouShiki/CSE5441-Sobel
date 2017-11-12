/***
 * File: maxwell_griffin_lab4p2.cu
 * Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
 *       serial algorithm, and once by a massively parallel CUDA algorithm.
 */

 #include <stdio.h>
 #include <Sobel.h>

 /*
  * Main function.
  */
 int main()
 {
    double x[3][3] = {{3,2,1}, {6,5,4}, {9,8,7}};
    Stencil_t stencil;
    stencil.top = x[0];
    stencil.middle = x[1];
    stencil.bottom = x[2];

    double result = Sobel_Magnitude(stencil);

    printf("The result of the edge detection is %lf", result);
 }
