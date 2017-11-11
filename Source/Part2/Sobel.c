/***
 * File: Sobel.c
 */

#include <math.h>
#include "Sobel.h"

static const Stencil_t Gx = {
   .top =    { -1, 0, 1 },
   .middle = { -2, 0, 2 },
   .bottom = { -1, 0, 1 }
}

static const Stencil_t Gy = {
   .top =    {  1,  2,  1 },
   .middle = {  0,  0,  0 },
   .bottom = { -1, -2, -1 }
}

double Sobel_Magnitude(Stencil_t *stencil)
{
   double sumGx = 0, sumGy = 0;

   for(int i = 0; i < 3; i++)
   {
      sumGx += (Gx.top[i] * stencil.top[i])
            +  (Gx.middle[i] * stencil.middle[i])
            +  (Gx.bottom[i] * stencil.bottom[i]);

      sumGy += (Gy.top[i] * stencil.top[i])
            +  (Gy.middle[i] * stencil.middle[i])
            +  (Gy.bottom[i] * stencil.bottom[i]);
   }

   return sqrt(pow(sumGx, 2) + pow(sumGy, 2));
}
