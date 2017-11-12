/***
 * File: Sobel.c
 */

#include <math.h>
#include "Sobel.h"

static const double Gx_data[3][3] = {
   { -1, 0, 1 },
   { -2, 0, 2 },
   { -1, 0, 1 },
}
static const Stencil_t Gx = {
   .top = Gx_data[0],
   .middle = Gx_data[1],
   .bottom = Gx_data[2]
};

static const double Gy_data[3][3] = {
   {  1,  2,  1 },
   {  0,  0,  2 },
   { -1, -2, -1 },
}
static const Stencil_t Gy = {
   .top = Gy_data[0],
   .middle = Gy_data[1],
   .bottom = Gy_data[2]
}

double Sobel_Magnitude(Stencil_t *stencil)
{
   double Gx_sum = 0, Gy_sum = 0;

   for(int i = 0; i < 3; i++)
   {
      Gx_sum += (Gx.top[i] * stencil.top[i])
            +  (Gx.middle[i] * stencil.middle[i])
            +  (Gx.bottom[i] * stencil.bottom[i]);

      Gy_sum += (Gy.top[i] * stencil.top[i])
            +  (Gy.middle[i] * stencil.middle[i])
            +  (Gy.bottom[i] * stencil.bottom[i]);
   }

   return sqrt(pow(Gx_sum, 2) + pow(Gy_sum, 2));
}
