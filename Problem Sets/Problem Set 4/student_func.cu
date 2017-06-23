//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void counter(unsigned int* d_count0,
                unsigned int* d_count1,
                unsigned int* const d_inputVals,
                const unsigned int mask)
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int val = (d_inputVals[idx] >> mask) & 1;
  if(!val){ 
    d_count0[idx] = 1;
    d_count1[idx] = 0;
  }
  else {
    d_count1[idx] = 1;
    d_count0[idx] = 0;
  }
}

__global__ void preSum(unsigned int* d_in, unsigned int* d_out)
{
  unsigned int blockStart = blockIdx.x*blockDim.x;
  unsigned int idx = blockStart + threadIdx.x;
  unsigned int temp = d_in[idx];
  __syncthreads();
  for(int s = 1; s < blockDim.x; s <<=1){
      unsigned int value = 0;
      if(idx >= blockStart + s){
        value = d_in[idx - s];
      }
      __syncthreads();
      d_in[idx] += value;
      __syncthreads();
  }
  if(idx == blockStart + blockDim.x - 1)
  {
    d_out[blockIdx.x] = d_in[idx]; 
    //printf("%d %d %d\n",idx,blockStart,blockIdx.x);
  }
  d_in[idx] -= temp;
}

__global__ void assignValue(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               unsigned int* const d_skip,
               unsigned int* const d_grid0,
               unsigned int* const d_grid1,
               unsigned int* const d_count0,
               unsigned int* const d_count1,
               const unsigned int mask)
{
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int val = (d_inputVals[idx] >> mask) & 1;
  unsigned int next_idx;
  if(val)
    next_idx = d_skip[0] + d_grid1[blockIdx.x] + d_count1[idx];
  else
    next_idx = d_grid0[blockIdx.x] + d_count0[idx];
  d_outputVals[next_idx] = d_inputVals[idx];
  d_outputPos[next_idx]  = d_inputPos[idx];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  const unsigned int gridSize = 320;
  const unsigned int blockSize = 689;

  unsigned int *d_skip;
  unsigned int *d_count0,*d_count1;
  unsigned int *d_grid0, *d_grid1 ;
  checkCudaErrors(cudaMalloc(&d_count0, sizeof(unsigned int)*numElems));
  checkCudaErrors(cudaMalloc(&d_count1, sizeof(unsigned int)*numElems));
  checkCudaErrors(cudaMalloc(&d_grid0, sizeof(unsigned int)*gridSize));
  checkCudaErrors(cudaMalloc(&d_grid1, sizeof(unsigned int)*gridSize));
  checkCudaErrors(cudaMalloc(&d_skip, sizeof(unsigned int)*2));
  
  checkCudaErrors(cudaMemcpy(d_outputPos , d_inputPos , sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));

  for (unsigned int i = 0; i < 32; i ++) {
    checkCudaErrors(cudaMemcpy(d_inputPos , d_outputPos , sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));

    counter <<<gridSize, blockSize>>> (d_count0, d_count1, d_inputVals, i);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    preSum <<<gridSize, blockSize>>> (d_count0, d_grid0);
    preSum <<<1, gridSize>>> (d_grid0, d_skip);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    preSum <<<gridSize, blockSize>>> (d_count1, d_grid1);
    preSum <<<1, gridSize>>> (d_grid1, d_skip + 1);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    assignValue<<<gridSize, blockSize>>> (d_inputVals,
                                          d_inputPos,
                                          d_outputVals,
                                          d_outputPos,
                                          d_skip,
                                          d_grid0,
                                          d_grid1,
                                          d_count0,
                                          d_count1,i); 
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  }

}
