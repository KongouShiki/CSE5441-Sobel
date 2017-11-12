/***
 * File: maxwell_griffin_lab4p2.cu
 * Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
 *       serial algorithm, and once by a massively parallel CUDA algorithm.
 */

#include <stdio.h>
#include <math.h>
#include "Stencil.h"

static double Gx_data[3][3] = {
   { -1, 0, 1 },
   { -2, 0, 2 },
   { -1, 0, 1 }
};
static Stencil_t Gx = {
   .top = Gx_data[0],
   .middle = Gx_data[1],
   .bottom = Gx_data[2]
};

static double Gy_data[3][3] = {
   {  1,  2,  1 },
   {  0,  0,  2 },
   { -1, -2, -1 }
};
static Stencil_t Gy = {
   .top = Gy_data[0],
   .middle = Gy_data[1],
   .bottom = Gy_data[2]
};

double Sobel_Magnitude(Stencil_t *stencil)
{
   double Gx_sum = 0, Gy_sum = 0;

   int i;
   for(i = 0; i < 3; i++)
   {
      Gx_sum += (Gx.top[i] * stencil->top[i])
       +  (Gx.middle[i] * stencil->middle[i])
       +  (Gx.bottom[i] * stencil->bottom[i]);

      Gy_sum += (Gy.top[i] * stencil->top[i])
       +  (Gy.middle[i] * stencil->middle[i])
       +  (Gy.bottom[i] * stencil->bottom[i]);
   }

   return sqrt(pow(Gx_sum, 2) + pow(Gy_sum, 2));
}

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

   double result = Sobel_Magnitude(&stencil);

   printf("The result of the edge detection is %lf\n", result);
}
