/***
 * File: maxwell_griffin_lab4p1.cu
 * Desc: Multiplies a matrix with the transpose of itself, as an exercise in
 *       writing a simple CUDA kernel and comparing it with its complementary
 *       serial implementation
 */

#include <stdbool>
#include <stdlib.h>
#include <time.h>

#define MATRIX_DIM 1024

/*
 * Sets values in an x by y matrix to random values between 1.0 and 2.0
 */
void InitMatrixToRandomValues(double *matrix, int dimX, int dimY)
{
   // see the RNG on first call to this function only
   static bool seeded = false;
   if(!seeded)
   {
      seeded = true;
      srand(time(NULL));
   }

   for(int y = 0; y < dimY; y++)
      for(int x = 0; x < dimX; x++)
         matrix[(y * dimY) + x] = ((double)rand() / RAND_MAX) + 1.0;
}

int main()
{
   /*
    * Serial algorithm to multiply the matrix with its transpose
    */

   double A[MATRIX_DIM][MATRIX_DIM];
   double C[MATRIX_DIM][MATRIX_DIM] = {{0}};    // static initialize to 0

   InitMatrixToRandomValues(A, MATRIX_DIM, MATRIX_DIM);

   for(int i = 0; i < MATRIX_DIM; i++)
      for(int j = 0; j < MATRIX_DIM; j++)
         for(int k = 0; k < MATRIX_DIM; k++)
            C[i][j] += A[k][i] * A[k][j]

   return 0;
}
