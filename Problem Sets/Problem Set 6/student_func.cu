//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the F, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
            else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>

__global__ void findBorder(const uchar4* const d_sourceImg,
                            unsigned int* d_count,
                            float* d_red,
                            float* d_green,
                            float* d_blue)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(d_sourceImg[idx].x != 255 ||  d_sourceImg[idx].y != 255 || d_sourceImg[idx].z != 255){
        if(threadIdx.x >= 1) atomicAdd(&d_count[idx - 1], 1);
        if(threadIdx.x + 1 < blockDim.x) atomicAdd(&d_count[idx + 1], 1);
        if(blockIdx.x  >= 1) atomicAdd(&d_count[idx - blockDim.x], 1);
        if(blockIdx.x + 1 < gridDim.x) atomicAdd(&d_count[idx + blockDim.x], 1);
        atomicAdd(&d_count[idx], 1);
    }
    d_red[idx] = d_sourceImg[idx].x;
    d_green[idx] = d_sourceImg[idx].y;
    d_blue[idx] = d_sourceImg[idx].z;
}
__global__ void initialize(const uchar4* const d_sourceImg,
                            const uchar4* const d_destImg,
                            unsigned int* d_count,
                            float* d_red,
                            float* d_green,
                            float* d_blue,
                            float* d_dif_red,
                            float* d_dif_green,
                            float* d_dif_blue)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(d_sourceImg[idx].x != 255 ||  d_sourceImg[idx].y != 255 || d_sourceImg[idx].z != 255){
        if(d_count[idx] == 5){
            d_dif_red[idx] = 4.f*d_sourceImg[idx].x - d_sourceImg[idx - 1].x
                                                     - d_sourceImg[idx + 1].x
                                                     - d_sourceImg[idx - blockDim.x].x
                                                     - d_sourceImg[idx + blockDim.x].x;
            d_dif_green[idx] = 4.f*d_sourceImg[idx].y - d_sourceImg[idx - 1].y
                                                       - d_sourceImg[idx + 1].y
                                                       - d_sourceImg[idx - blockDim.x].y
                                                       - d_sourceImg[idx + blockDim.x].y;
            d_dif_blue[idx] = 4.f*d_sourceImg[idx].z - d_sourceImg[idx - 1].z
                                                      - d_sourceImg[idx + 1].z
                                                      - d_sourceImg[idx - blockDim.x].z
                                                      - d_sourceImg[idx + blockDim.x].z;
        }else{
            d_red[idx] = d_destImg[idx].x;
            d_green[idx]= d_destImg[idx].y;
            d_blue[idx] = d_destImg[idx].z;
        }
    }
}
__global__ void run(const uchar4* const d_sourceImg,
                    unsigned int* d_count,
                    float* d_prev_r,
                    float* d_prev_g,
                    float* d_prev_b,
                    float* d_next_r,
                    float* d_next_g,
                    float* d_next_b,
                    float* d_dif_r,
                    float* d_dif_g,
                    float* d_dif_b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(d_count[idx] == 5){
        d_next_r[idx] = min(255.f,max(0.f, ( d_dif_r[idx]
                                             + d_prev_r[idx - 1]
                                             + d_prev_r[idx + 1]
                                             + d_prev_r[idx - blockDim.x]
                                             + d_prev_r[idx + blockDim.x])/4.f));    
        d_next_g[idx] = min(255.f,max(0.f, ( d_dif_g[idx]
                                             + d_prev_g[idx - 1]
                                             + d_prev_g[idx + 1]
                                             + d_prev_g[idx - blockDim.x]
                                             + d_prev_g[idx + blockDim.x])/4.f));
        d_next_b[idx] = min(255.f,max(0.f, ( d_dif_b[idx]
                                             + d_prev_b[idx - 1]
                                             + d_prev_b[idx + 1]
                                             + d_prev_b[idx - blockDim.x]
                                             + d_prev_b[idx + blockDim.x])/4.f));
    }
    else{
        d_next_r[idx] = d_prev_r[idx];
        d_next_g[idx] = d_prev_g[idx];
        d_next_b[idx] = d_prev_b[idx];
    }
}

__global__ void finalize(uchar4* d_blendedImg,
                        uchar4* d_destImg,
                        unsigned int* d_count,
                        float* d_red,
                        float* d_green,
                        float* d_blue)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(d_count[idx] == 5){
        d_blendedImg[idx].x = d_red[idx];
        d_blendedImg[idx].y = d_green[idx];
        d_blendedImg[idx].z = d_blue[idx];
        
    }else{
        d_blendedImg[idx] = d_destImg[idx];
    }
}
void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    uchar4* d_sourceImg;
    uchar4* d_destImg;
    uchar4* d_blendedImg;
    unsigned int* d_count;
    float *d_dif_r,*d_dif_g,*d_dif_b;
    float *d_prev_r,*d_prev_g,*d_prev_b;
    float *d_next_r,*d_next_g,*d_next_b;

    checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4)*numRowsSource*numColsSource));    
    checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(uchar4)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_destImg, sizeof(uchar4)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned int)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_dif_r, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_dif_g, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_dif_b, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_prev_r, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_prev_g, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_prev_b, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_next_r, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_next_g, sizeof(float)*numRowsSource*numColsSource));
    checkCudaErrors(cudaMalloc(&d_next_b, sizeof(float)*numRowsSource*numColsSource));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemset(d_count, 0, sizeof(unsigned int)*numRowsSource*numColsSource));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    findBorder <<<numRowsSource,numColsSource>>> (d_sourceImg, d_count, d_prev_r, d_prev_g, d_prev_b);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    initialize <<<numRowsSource,numColsSource>>> (d_sourceImg, d_destImg, d_count, d_prev_r, d_prev_g, d_prev_b, d_dif_r, d_dif_g, d_dif_b);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for(int i=0;i < 800; i++){
        run <<<numRowsSource,numColsSource>>> (d_sourceImg, d_count, d_prev_r, d_prev_g, d_prev_b, d_next_r, d_next_g, d_next_b, d_dif_r, d_dif_g, d_dif_b);    
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        std::swap(d_next_r, d_prev_r);
        std::swap(d_next_g, d_prev_g);
        std::swap(d_next_b, d_prev_b);
    }
    finalize <<<numRowsSource,numColsSource>>> (d_blendedImg, d_destImg, d_count, d_prev_r, d_prev_g, d_prev_b);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4)*numRowsSource*numColsSource, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
}
