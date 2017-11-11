#include <math.h>
#include "Sobel.h"

typedef struct
{
   double Gx[3][3];
   double Gy[3][3]
} SobelGradient_t;

static const SobelGradient_t gradient = {
   .Gx = {
      { -1, 0, 1 },
      { -2, 0, 2 },
      { -1, 0, 1 },
   },

   .Gy = {
      { 1, 2, 1 },
      { 0, 0, 0 },
      { -1, -2, -1 },
   }
}

double Sobel_Magnitude(Stencil_t *stencil)
{
   double Gx = 0, Gy = 0;

   for(int i = 0; i < 3; i++)
   {
      Gx += (gradient.Gx[0][i] * stencil.top[i])
         +  (gradient.Gx[1][i] * stencil.middle[i])
         +  (gradient.Gx[2][i] * stencil.bottom[i]);

      Gy += (gradient.Gy[0][i] * stencil.top[i])
         +  (gradient.Gy[1][i] * stencil.middle[i])
         +  (gradient.Gy[2][i] * stencil.bottom[i]);
   }

   return sqrt(pow(Gx, 2) + pow(Gy, 2));
}
