/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <cstdio>
#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals,
               unsigned int* const histo)
{
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  __shared__ unsigned int histogram[1024]
  histogram[threadIdx.x] = 0;
  __syncthreads();
  atomicAdd(&histogram[vals[idx]],1);
  __syncthreads();
  atomicAdd(&histo[threadIdx.x],histogram[threadIdx.x]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  yourHisto <<<numElems/numBins,numBins>>> (d_vals,d_histo);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
