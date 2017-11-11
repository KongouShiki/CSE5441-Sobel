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

double Sobel_Magnitude(double stencil[3][3])
{
   double Gx = 0, Gy = 0;

   for(int i = 0; i < 3l i++)
   {
      for(int j = 0; j < 3; j++)
      {
         Gx += gradient.Gx[i][j] * stencil[i][j];
         Gy += gradient.Gy[i][j] * stencil[i][j];
      }
   }

   return sqrt(pow(Gx, 2) + pow(Gy, 2));
}
