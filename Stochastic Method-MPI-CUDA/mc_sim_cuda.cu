#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include "time.h"


// CUDA libraries
#include <cuda.h>
#include <curand.h>          //CUDA random number library
#include <curand_kernel.h>


const int N = 10240;

// Seed the random number generator (call only once)
__global__ void setup_kernel( curandState * state, unsigned long seed )
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init ( seed, id, 0, &state[id] );
}

__global__ void mc_sim( curandState* globalState, int *win_a, int *win_b )
{

  // printf("block coordinates are is (%d , %d) \n", blockIdx.x, blockIdx.y);
  // printf("thread coordinates are is (%d , %d) \n", threadIdx.x, threadIdx.y);
  // printf("block dimensions are is (%d , %d) \n", blockDim.x, blockDim.y);

  int ind = threadIdx.x + blockDim.x * blockIdx.x;

  if ( ind < N ) {
    int count = 4;
    int i;
    int Door[4] = {0};
    int Door_rem[2] = {0}; // The two remaining doors for choice b
    int door_win;
    int door_choice;
    int door_choice_b;
    int elim_door;

    curandState localState = globalState[ind];
    // curand_uniform gives random number in (0,1) with uniform probability
    // you could change this call to curand_normal to get normal distribution
    // float RANDOM = curand_uniform( &localState );
    // Ad[ind] = curand_uniform( &localState );
    globalState[ind] = localState;

    // float r = unidev();
    float r = curand_uniform( &localState );
    // printf("r = %f \n", r);

    if (r < 0.25) {
      Door[0] = 1;
      door_win = 1;
    }
    else if (r > 0.25 && r < 0.5) {
      Door[1] = 1;
      door_win = 2;
    }
    else if ( r > 0.5 && r < 0.75) {
      Door[2] = 1;
      door_win = 3;
    }
    else  {
      Door[3] = 1;
      door_win = 4;
    }

    float m = curand_uniform( &localState );
    // printf("m = %f \n", m);

    if (m < 0.25) {
      door_choice = 1;
    }
    else if (m > 0.25 && m < 0.5) {
      door_choice = 2;
    }
    else if ( m > 0.5 && m < 0.75) {
      door_choice = 3;
    }
    else  {
      door_choice = 4;
    }

    for (i = 0; i < count; ++i)
    {
      if (Door[i] == 0 && door_choice != (i + 1))  {
        elim_door = i + 1;
        break;
      }
    }

    int j = 0;
    for (i = 0; i < count; ++i)
    {
      if (elim_door != (i + 1) && door_choice != (i + 1))
      {
        Door_rem[j] = (i + 1);
        j++;
      }
    }


    // Choice b
    float b_choice = curand_uniform( &localState );
    if (b_choice < 0.5) {
      door_choice_b = Door_rem[0];
    }
    else  {
      door_choice_b = Door_rem[1];
    }

    __syncthreads();

    // // Choice a
    if (door_win == door_choice)
      win_a[ind] = 1;

    // Choice b
    if (door_win == door_choice_b)
      win_b[ind] = 1;
  }
}

template <typename T> // this is the template parameter declaration
T sum(T *a, int N)
{
  T sum = 0;
  for (int i = 0; i < N; ++i)
  {
    sum += a[i];
  }
  return sum;
}

int main(int argc, char** argv)
{
  int *win_a;
  int *win_b;
  int group_n[9] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  char str[20];
  FILE *fp;

  for (int l = 0; l < 2; ++l)
  {
    // int l = 2;
    printf("Sample size is  l = %d \n", l);

    int trial_num = N / group_n[l];
    size_t size = sizeof(int) * trial_num;

    // int list_length = trial_num / group_n[0];
    float *Pa_array;
    float *Pb_array;

    // Device variables
    int *win_a_d;
    int *win_b_d;

    /* initialize random number generator */
    // srandom(seed);

    int blockdimx = 1024;
    int griddimx = ceil(trial_num / blockdimx) + 1;

    Pa_array = (float *)calloc(group_n[l], sizeof(float));
    Pb_array = (float *)calloc(group_n[l], sizeof(float));

    // int blockdimx = trial_num;
    // int griddimx = (trial_num / blockdimx < 1) ? 1 : ceil(trial_num / blockdimx);

    // printf("blockdimx = %d \n", blockdimx);
    // printf("griddimx = %d \n", griddimx);

    dim3 block(blockdimx, 1);
    dim3 grid(griddimx, 1);
    curandState* devStates;

    // Monte Carlo simulation
    for (int i = 0; i < group_n[l]; ++i)
    {

      int seed = (int)time(NULL) * i * l;
      win_a = (int *)calloc(trial_num, sizeof(int));
      win_b = (int *)calloc(trial_num, sizeof(int));

      cudaMalloc((void **)&win_a_d, size);
      cudaMalloc((void **)&win_b_d, size);
      cudaMalloc(&devStates, trial_num * sizeof( curandState ) );

      cudaMemcpy(win_a_d, win_a, size, cudaMemcpyHostToDevice);
      cudaMemcpy(win_b_d, win_b, size, cudaMemcpyHostToDevice);

      // setup seeds.  Note the 2nd argument is the seed which
      // you can set to anything you like
      // setup_kernel <<< 1, block >>> ( devStates, 1234678 );
      setup_kernel <<< grid, block >>> ( devStates, seed );

      printf("Starting the MC simulations \n");


      // int seed = 34569;
      mc_sim <<< grid, block >>> ( devStates, win_a_d, win_b_d );

      cudaDeviceSynchronize();

      cudaMemcpy(win_a, win_a_d, size, cudaMemcpyDeviceToHost);
      cudaMemcpy(win_b, win_b_d, size, cudaMemcpyDeviceToHost);

      printf("\nNumber of trials was = %d \n", trial_num);

      int sum_a = sum(win_a, trial_num);
      int sum_b = sum(win_b, trial_num);

      float pa = (float)sum_a / (float)trial_num;
      float pb = (float)sum_b / (float)trial_num;

      Pa_array[i] = pa;
      Pb_array[i] = pb;

      printf("Choice A won %d times \n", sum_a);
      printf("Pa = %f \n", pa);
      printf("Choice B won %d times \n", sum_b);
      printf("Pb = %f \n", pb);

      for (int s = 0; s < trial_num; ++s)
      {
        win_a[s] = 0;
        win_b[s] = 0;
      }


    }


    printf("\nCleaning up... \n");

    int k;

    // Output results
    sprintf(str, "p_a_n%d.txt", group_n[l]);
    fp = fopen(str, "wt");
    for (k = 0; k < group_n[l]; k++) {
      fprintf(fp, " %f \n", Pa_array[k]);
    }
    fclose(fp);

// Output results
    sprintf(str, "p_b_n%d.txt", group_n[l]);
    fp = fopen(str, "wt");
    for (k = 0; k < group_n[l]; k++) {
      fprintf(fp, " %f \n", Pb_array[k]);
    }
    fclose(fp);

    // GPU
    cudaFree(&win_a_d);
    cudaFree(&win_b_d);
    cudaFree(devStates);

    // CPU
    free(win_a);
    free(win_b);
    free(Pa_array);
    free(Pb_array);

  }

}

