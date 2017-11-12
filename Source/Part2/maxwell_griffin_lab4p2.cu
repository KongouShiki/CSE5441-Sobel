/***
 * File: maxwell_griffin_lab4p2.cu
 * Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
 *       serial algorithm, and once by a massively parallel CUDA algorithm.
 */

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "Stencil.h"

#define LINEARIZE(row, col, dim) \
   (((row) * (dim)) + (col))

/*
 * Globals
 */
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


/*
 * Calculates the magnitude of the gradient of the stencil of image values,
 * following the Sobel edge detection operator
 *
 * @param stencil -- the 3x3 image stencil
 * @return -- magnitude of the gradient
 */
double SobelPixelMagnitude(Stencil_t *stencil)
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
int main(int argc, char* argv[])
{
   // Check for correct number of comand line args
   if (argc != 4)
   {
      printf("Error: Incorrect arguments.\n");
      printf("Should be: <input.bmp> <serial_output.bmp> <cuda_output.bmp>\n");
      return 0;
   }

   // Open the files specified by the command line args
   FILE *inputFile = fopen(argv[1], "rb");
   FILE *serialOutputFile = fopen(argv[2], "wb");
   FILE *cudaOutputFile = fopen(argv[3], "wb");
   if(inputFile == NULL)
   {
      printf("Error: %s could not be opened for reading.", argv[1]);
   }

   // Read in input image and allocate space for new output image buffers
   uint8_t *inputImage = (uint8_t *)read_bmp_file(inFile);
	uint8_t *serialOutputImage = (uint8_t *)malloc(get_num_pixel());
	uint8_t *cudaOutputImage = (uint8_t *)malloc(get_num_pixel());

   // Free allocated memory
   free(serialOutputImage);
   free(cudaOutputImage);
}
