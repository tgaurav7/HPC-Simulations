// Compiled with
// nvcc ./GPUJacobi.cu
// Run on monk with
// sqsub -q gpu --gpp=1 -r 10m -o CUDA_TEST ./a.out
// Get job status via
// sqjobs
// state = R means it is running
// state = D means it is done

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


const int BLOCK_SIZE_X = 18;  //interior + 2 boundary layers, ie. 16x16 + boundary layers=18x18
const int BLOCK_SIZE_Y = 18;

__global__ void JacobiRelaxationGPU(float* u_d, float* f_d, int ArraySizeX, int ArraySizeY, float h)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x*(BLOCK_SIZE_X-2);
  int by=blockIdx.y*(BLOCK_SIZE_Y-2);
  int x=tx+bx;
  int y=ty+by;

  __shared__ float u_sh[BLOCK_SIZE_X][BLOCK_SIZE_Y];

  u_sh[tx][ty]=u_d[x+y*ArraySizeX];
 
  __syncthreads();

  // Interior threads only do actual work below.  Note that the 0.25f is not a typo, it means
  // that this constant should be interpreted as a float, not a double (the default in c/c++)
  if(tx>0 && tx<BLOCK_SIZE_X-1 && ty>0 && ty<BLOCK_SIZE_Y-1) {
    u_d[x+y*ArraySizeX] = 0.25f*(u_sh[tx+1][ty]+u_sh[tx-1][ty]+u_sh[tx][ty+1]+u_sh[tx][ty-1]-h*h*f_d[x+y*ArraySizeX]);
  }
}

int main(void)
{
  float *u_h, *f_h;    // pointers to host memory
  float *u_d, *f_d;    // pointers to device memory
  int ArraySizeX=1026;  // Note these, minus boundary layer, have to be exactly divisible by the (BLOCK_SIZE-2) here
  int ArraySizeY=1026;
  size_t size=ArraySizeX*ArraySizeY*sizeof(float);
  FILE *fp;

  //Allocate arrays on host and initialize to zero
  u_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  f_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));

  //Allocate arrays on device
  cudaMalloc((void **) &u_d,size);
  cudaMalloc((void **) &f_d,size);

  //Initialize arrays u_h and f_h boundaries
    /* a particularly boring set of boundary conditions */
    //for(int i=0; i < ArraySizeY; i++)
    //  u_h[i*ArraySizeX] = u_h[i*ArraySizeX+ArraySizeX-1] =1.0;
    //for(int j=1; j< ArraySizeX; j++) 
    //  u_h[j] = u_h[(ArraySizeY-1)*ArraySizeX+j]=1.0;
    for(int i=ArraySizeY/4; i<3*ArraySizeY/4; i++) {
      f_h[ArraySizeY/4*ArraySizeX+i]=0.25;
      f_h[ArraySizeY/4*3*ArraySizeX+i]=-0.25;
    }

  //Perform computation on GPU
  // Part 1 of 4: Copy data from host to device
  cudaMemcpy(u_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f_h, size, cudaMemcpyHostToDevice);

  // Part 2 of 4: Set up execution configuration
  int nBlocksX=(ArraySizeX-2)/(BLOCK_SIZE_X-2);
  int nBlocksY=(ArraySizeY-2)/(BLOCK_SIZE_Y-2);

  dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 dimGrid(nBlocksX,nBlocksY);


  for (int nsteps=1; nsteps < 200000; nsteps++) {
    // Part 3 of 4: Call kernel with execution configuration
    JacobiRelaxationGPU<<<dimGrid, dimBlock>>>(u_d,f_d,ArraySizeX,ArraySizeY, 1.0);
  }   


  // Part 4 of 4: Retrieve result from device and store in u_h
  cudaMemcpy(u_h, u_d, size, cudaMemcpyDeviceToHost);

  // Output results
  fp = fopen("Solution.txt","wt");
  for (int i=0; i<ArraySizeX; i++) {
    for (int j=0; j<ArraySizeY; j++)
      fprintf(fp," %f",u_h[j*ArraySizeX+i]);
    fprintf(fp,"\n");
  }
  fclose(fp);

  //Cleanup
  free(u_h);
  free(f_h);
  cudaFree(u_d);
  cudaFree(f_d);
}
