/***
 * File: maxwell_griffin_lab4p1.cu
 * Desc: Multiplies a matrix with the transpose of itself, as an exercise in
 *       writing a simple CUDA kernel and comparing it with its complementary
 *       serial implementation
 */

#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_DIM 1024

/*
 * Sets values in an x by y matrix to random values between 1.0 and 2.0
 * @param matrix -- a matrix
 * @param rows -- number of rows in matrix
 * @param cols -- number of columns in matrix
 */
void InitMatrixToRandomValues(double *matrix, int rows, int cols)
{
   // seed the RNG on first call to this function only
   static bool seeded = false;
   if(!seeded)
   {
      seeded = true;
      srand(time(NULL));
   }

   for(int col = 0; col < cols; col++)
      for(int row = 0; row < rows; row++)
         matrix[(col * cols) + row] = ((double)rand() / RAND_MAX) + 1.0;
}

/*
 * Serial algorithm to multiply a square matrix with its transpose
 * @param inputMatrix -- the matrix to perform the multiplication on
 * @param resultMatrix -- stores the result
 * @param dim -- the size of each row and column
 */
void SerialMultiplyMatrixByTranspose(
   double *inputMatrix,
   double *resultMatrix,
   int dim)
{
   for(int i = 0; i < dim; i++)
      for(int j = 0; j < dim; j++)
         for(int k = 0; k < dim; k++)
            resultMatrix[i][j] += A[k][i] * A[k][j];
}

int main()
{
   double A_serial[MATRIX_DIM][MATRIX_DIM];
   static double C_serial[MATRIX_DIM][MATRIX_DIM];    // static initialize to 0

   InitMatrixToRandomValues(&A_serial[0][0], MATRIX_DIM, MATRIX_DIM);
   SerialMultiplyMatrixByTranspose(A_serial, C_serial, MATRIX_DIM);

   return 0;
}
