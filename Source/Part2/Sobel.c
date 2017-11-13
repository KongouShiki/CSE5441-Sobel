/***
 * File: Sobel.c
 */

#include <math.h>
#include "Sobel.h"

static int Gx[3][3] = {
   { -1, 0, 1 },
   { -2, 0, 2 },
   { -1, 0, 1 }
};

static int Gy[3][3] = {
   {  1,  2,  1 },
   {  0,  0,  0 },
   { -1, -2, -1 }
};

double Sobel_Magnitude(Stencil_t *stencil)
{
   double sumGx = 0, sumGy = 0;

   int i;
   for(i = 0; i < 3; i++)
   {
      sumGx += (Gx[0][i] * stencil->top[i])
         +  (Gx[1][i] * stencil->middle[i])
         +  (Gx[2][i] * stencil->bottom[i]);

      sumGy += (Gy[0][i] * stencil->top[i])
         +  (Gy[1][i] * stencil->middle[i])
         +  (Gy[2][i] * stencil->bottom[i]);
   }

   return sqrt(pow(sumGx, 2) + pow(sumGy, 2));
}
