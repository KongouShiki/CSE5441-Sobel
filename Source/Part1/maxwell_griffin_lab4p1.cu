/***
 * File: maxwell_griffin_lab4p1.cu
 * Desc: Multiplies a matrix with the transpose of itself, as an exercise in
 *       writing a simple CUDA kernel and comparing it with its complementary
 *       serial implementation
 */

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_DIM 1024

#define LINEARIZE(row, col, dim) \
   (((row) * (dim)) + (col))

/*
 * Sets values in an x by y matrix to random values between 1.0 and 2.0
 *
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
         matrix[LINEARIZE(row, col, cols)] = ((double)rand() / RAND_MAX) + 1.0;
}

/*
 * Serial algorithm to multiply a square matrix with its transpose
 *
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
            resultMatrix[LINEARIZE(i, j, dim)] += inputMatrix[LINEARIZE(k, i, dim)] * inputMatrix[LINEARIZE(k, j, dim)];
}


/*
 * Massively parallel CUDA kernel function that multiplies a matrix with its transpose.
 * Each thread calculates an entire row of the result matrix.
 *
 * @param inputMatrix -- the device matrix to perform the multiplication on
 * @param resultMatrix -- device memory to store the result
 * @param dim -- the size of each row and column
 */
__global__ void cudaMultiplyMatrixByTranspose(
   double *inputMatrix,
   double *resultMatrix,
   int dim)
{
   int i = threadIdx.x; // Row of result and col of input
   int j;               // Col of result and col of transpose
   int k;               // Row of input  and row of transpose
   double sum = 0;


   for(j = 0; j < dim; j++)
   {
      sum = 0;
      for(k = 0; k < dim; k++)
      {
         sum += inputMatrix[LINEARIZE(k, i, dim)] * inputMatrix[LINEARIZE(k, j, dim)];
      }

      resultMatrix[LINEARIZE(i, j, dim)] = sum;
   }
}

/*
 * Parallel algorithm that uses CUDA to multiply a square matrix with its transpose.
 * Grid, block, and thread dimensions picked to be optimal for a Tesla P100 GPU.
 *
 * @param inputMatrix -- the matrix to perform the multiplication on
 * @param resultMatrix -- stores the result
 * @param dim -- the size of each row and column
 */
void ParallelMultiplyMatrixByTranspose(
   double *inputMatrix,
   double *resultMatrix,
   int dim)
{
   double *deviceInputMatrix, *deviceResultMatrix;
   int numBlocks = 1;   // One block to start
   int threadsPerBlock = 1024;
   size_t matrixMemSize = dim * dim * sizeof(double);

   // Allocate device memory
   cudaMalloc((void **)&deviceInputMatrix, matrixMemSize);
   cudaMalloc((void **)&deviceResultMatrix, matrixMemSize);

   // Copy host input array to device
   cudaMemcpy(deviceInputMatrix, inputMatrix, matrixMemSize, cudaMemcpyHostToDevice);

   // Launch Kernel
   dim3 dimGrid(numBlocks);
   dim3 dimBlock(threadsPerBlock);
   cudaMultiplyMatrixByTranspose<<<dimGrid, dimBlock>>>(deviceInputMatrix, deviceResultMatrix, dim);

   // Copy device results array back to host
   cudaMemcpy(resultMatrix, deviceResultMatrix, matrixMemSize, cudaMemcpyDeviceToHost);
}

int main()
{
   double A_serial[MATRIX_DIM][MATRIX_DIM], A_parallel[MATRIX_DIM][MATRIX_DIM];
   static double C_serial[MATRIX_DIM][MATRIX_DIM], C_parallel[MATRIX_DIM][MATRIX_DIM];    // static initialize to 0

   InitMatrixToRandomValues(&A_serial[0][0], MATRIX_DIM, MATRIX_DIM);
   SerialMultiplyMatrixByTranspose(A_serial, C_serial, MATRIX_DIM);

   InitMatrixToRandomValues(&A_parallel[0][0], MATRIX_DIM, MATRIX_DIM);
   ParallelMultiplyMatrixByTranspose(A_parallel, C_parallel, MATRIX_DIM);

   if(0 == memcmp(C_serial, C_parallel, MATRIX_DIM * MATRIX_DIM * sizeof(double)))
   {
      printf("The two resulting matrices are the same.\n");
   }
   else
   {
      printf("The resulting matrices do not match!\n");
   }

   return 0;
}
