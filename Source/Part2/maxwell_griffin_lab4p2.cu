/***
 * File: maxwell_griffin_lab4p2.cu
 * Desc: Performs 2 Sobel edge detection operations on a .bmp, once by a
 *       serial algorithm, and once by a massively parallel CUDA algorithm.
 */

#include <stdio.h>
#include <math.h>
#include "Stencil.h"
#include "read_bmp.h"

#define PERCENT_BLACK_THRESHOLD 75

#define CUDA_GRIDS (1)
#define CUDA_BLOCKS_PER_GRID (1)
#define CUDA_THREADS_PER_BLOCK (1)

#define MS_PER_SEC (1000)
#define NS_PER_MS (1000 * 1000)
#define NS_PER_SEC (NS_PER_MS * MS_PER_SEC)

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

// Timing structs
static struct timespec rtcSerialStart;
static struct timespec rtcSerialEnd;
static struct timespec rtcParallelStart;
static struct timespec rtcParallelEnd;


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
 * Display all header information and matrix and CUDA parameters.
 */
void DisplayParameters(
   char *inputFile,
   char *serialOutputFile,
   char *cudaOutputFile)
{
   printf("********************************************************************************\n");
   printf("lab4p2: serial vs. CUDA Sobel edge detection.\n");
   printf("\n");
   printf("Input image: %s \t(Height: %d pixels, width: %d pixels)\n");
   printf("Serial output image: \t%s\n", serialOutputFile);
   printf("CUDA output image: \t%s\n", cudaOutputFile);
   printf("\n");
   printf("CUDA compute structure:\n");
   printf("|-- with %d grid\n", CUDA_GRIDS);
   printf("    |-- with %d blocks\n", CUDA_BLOCKS_PER_GRID);
   printf("        |-- with %d threads per block\n", CUDA_THREADS_PER_BLOCK);
   printf("\n");
}

/*
 * Display the timing and convergence results to the screen.
 *
 * @param serialConvergenceThreshold
 * @param serialConvergenceThreshold
 */
void DisplayResults(
   int serialConvergenceThreshold,
   int parallelConvergenceThreshold)
{
   printf("Time taken for serial Sobel edge detection: %lf\n",
      (LINEARIZE(rtcSerialEnd.tv_sec, rtcSerialEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcSerialStart.tv_sec, rtcSerialStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));

   printf("Convergence Threshold: %d\n", serialConvergenceThreshold);
   printf("\n");

   printf("Time taken for serial Sobel edge detection: %lf\n",
      (LINEARIZE(rtcParallelEnd.tv_sec, rtcParallelEnd.tv_nsec, NS_PER_SEC)
      - LINEARIZE(rtcParallelStart.tv_sec, rtcParallelStart.tv_nsec, NS_PER_SEC))
      / ((double)NS_PER_SEC));

   printf("Convergence Threshold: %d\n", parallelConvergenceThreshold);
   printf("********************************************************************************\n");
}

/*
 * Serial algorithm to keep perform a Sobel edge detection on an input pixel
 * buffer at different brightness thresholds until a certain percentage of
 * pixels in the output pixel buffer are black.
 *
 * @param input -- input pixel buffer
 * @param output -- output pixel buffer
 * @return -- brightness threshold at which PERCENT_BLACK_THRESHOLD pixels are black
 */
int SerialSobelEdgeDetection(uint8_t *input, uint8_t *output)
{
   return 100;
}

/*
 * Parallel algorithm to keep perform a Sobel edge detection on an input pixel
 * buffer at different brightness thresholds until a certain percentage of
 * pixels in the output pixel buffer are black.
 *
 * @param input -- input pixel buffer
 * @param output -- output pixel buffer
 * @return -- brightness threshold at which PERCENT_BLACK_THRESHOLD pixels are black
 */
int ParallelSobelEdgeDetection(uint8_t *input, uint8_t *output)
{
   return 20;
}

/*
* Main function.
*/
int main(int argc, char* argv[])
{
   // Check for correct number of comand line args
   if (argc != 4)
   {
      printf("Error: Incorrect arguments: <input.bmp> <serial_output.bmp> <cuda_output.bmp>\n");
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
   uint8_t *inputImage = (uint8_t *)read_bmp_file(inputFile);
	uint8_t *serialOutputImage = (uint8_t *)malloc(get_num_pixel());
	uint8_t *cudaOutputImage = (uint8_t *)malloc(get_num_pixel());

   DisplayParameters(argv[1], argv[2], argv[3]);

   printf("Performing serial Sobel edge detection.\n");
   clock_gettime(CLOCK_REALTIME, &rtcSerialStart);
   int serialConvergenceThreshold = SerialSobelEdgeDetection(inputImage, serialOutputImage);
   clock_gettime(CLOCK_REALTIME, &rtcSerialEnd);

   printf("Performing CUDA parallel Sobel edge detection.\n");
   clock_gettime(CLOCK_REALTIME, &rtcParallelStart);
   int parallelConvergenceThreshold = ParallelSobelEdgeDetection(inputImage, cudaOutputImage);
   clock_gettime(CLOCK_REALTIME, &rtcParallelEnd);

   DisplayResults(serialConvergenceThreshold, parallelConvergenceThreshold);

   // Write output image buffers
   write_bmp_file(serialOutputFile, serialOutputImage);
   write_bmp_file(cudaOutputFile, cudaOutputImage);

   // Close files
   fclose(serialOutputFile);
   fclose(cudaOutputFile);

   // Free allocated memory
   free(serialOutputImage);
   free(cudaOutputImage);
}
