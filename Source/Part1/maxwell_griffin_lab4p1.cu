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

#define MATRIX_DIM (1024)
#define THRESHOLD (0.00001)

#define CUDA_GRIDS (1)
#define CUDA_BLOCKS_PER_GRID (1024)
#define CUDA_THREADS_PER_BLOCK (1024)

#define MS_PER_SEC (1000)
#define NS_PER_MS (1000 * 1000)
#define NS_PER_SEC (NS_PER_MS * MS_PER_SEC)

#define LINEARIZE(row, col, dim) \
   (((row) * (dim)) + (col))

// Timing structs
static struct timespec rtcSerialStart;
static struct timespec rtcSerialEnd;
static struct timespec rtcParallelStart;
static struct timespec rtcParallelEnd;

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

   for(j = blockIdx.x; j < dim; j += gridDim.x)
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
   int numBlocks = CUDA_BLOCKS_PER_GRID;
   int threadsPerBlock = CUDA_THREADS_PER_BLOCK;
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

/*
 * Compares the values of two arrays of doubles to within some threshold.
 *
 * @param first -- an array of doubles
 * @param second -- another array of doubles
 * @param size -- the size of the arrays
 * @param theshold -- the maximum difference allowed between two values
 * @return size_t -- 0 if the arrays are equal
 *                   else where they first diverge, indexed at 1
 */
size_t CompareDoubleArrays(double *first, double *second, size_t size, double threshold)
{
   for(size_t i = 0; i < size; i++)
   {
      double diff = first[i] - second[i];
      if((diff > 0 && diff > threshold) || (diff < 0 && diff < (threshold * -1)))
      {
         return i+1;
      }
   }
   return 0;
}

/*
 * Display all header information and matrix and CUDA parameters.
 */
void DisplayParameters()
{
   printf("********************************************************************************\n");
   printf("lab4p1: serial vs. CUDA matrix multiplication.\n");
   printf("\n");
   printf("Matrix dimensions: %dx%d\n", MATRIX_DIM, MATRIX_DIM);
   printf("CUDA compute structure:\n");
   printf("|-- with %d grid\n", CUDA_GRIDS);
   printf("    |-- with %d blocks\n", CUDA_BLOCKS_PER_GRID);
   printf("        |-- with %d threads per block\n", CUDA_THREADS_PER_BLOCK);
   printf("\n");
}

/*
 * Compare the two resulting matrices and display the results to the screen.
 *
 * @param serial -- serial results matrix
 * @param parallel -- parallel  results matrix
 * @param dim -- row/col size of the matrices
 */
void DisplayResults(double *serial, double *parallel, int dim)
{
   printf("Checking that the two resulting matrices are equivalent.\n");
   size_t differ = CompareDoubleArrays(serial, parallel, dim * dim, THRESHOLD);
   if(0 == differ)
   {
      printf("The two resulting matrices are equivalent!\n", THRESHOLD);
   }
   else
   {
      size_t badRow = (differ - 1) / dim;
      size_t badCol = (differ - 1) % dim;
      printf("The resulting matrices do not match!\n");
      printf("The first non-matching values occur at (%d, %d).\n", badRow, badCol);
      printf("|-- Serial[%d][%d] = %d\n", badRow, badCol, serial[differ-1]);
      printf("|-- Parallel[%d][%d] = %d\n", badRow, badCol, parallel[differ-1]);
   }

   printf("\n");
   printf("Timing results:\n");
   printf("|-- The serial algorithm took %9.4lf seconds\n",
      (LINEARIZE(rtcSerialEnd.tv_sec, rtcSerialEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcSerialStart.tv_sec, rtcSerialStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));

   printf("|-- The parallel algorithm took %9.4lf seconds\n",
      (LINEARIZE(rtcParallelEnd.tv_sec, rtcParallelEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcParallelStart.tv_sec, rtcParallelStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));
}

/*
 * Main function.
 */
int main()
{
   double A[MATRIX_DIM][MATRIX_DIM];
   static double C_serial[MATRIX_DIM][MATRIX_DIM], C_parallel[MATRIX_DIM][MATRIX_DIM];    // static initialize to 0

   InitMatrixToRandomValues(&A[0][0], MATRIX_DIM, MATRIX_DIM);

   DisplayParameters();

   printf("Performing serial matrix multiplication.\n");
   clock_gettime(CLOCK_REALTIME, &rtcSerialStart);
   SerialMultiplyMatrixByTranspose(&A[0][0], &C_serial[0][0], MATRIX_DIM);
   clock_gettime(CLOCK_REALTIME, &rtcSerialEnd);

   printf("Performing parallel matrix multiplication.\n");
   clock_gettime(CLOCK_REALTIME, &rtcParallelStart);
   ParallelMultiplyMatrixByTranspose(&A[0][0], &C_parallel[0][0], MATRIX_DIM);
   clock_gettime(CLOCK_REALTIME, &rtcParallelEnd);

   DisplayResults(&C_serial[0][0], &C_parallel[0][0], MATRIX_DIM);

   return 0;
}
