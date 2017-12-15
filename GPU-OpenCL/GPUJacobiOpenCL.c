// Compiled with
// nvcc ./GPUJacobi.cu
// gcc GPUJacobiOpenCL.c -lOpenCL -I/opt/sharcnet/cuda/6.0.37/toolkit/include
// Run on monk with
// sqsub -q gpu --gpp=1 -r 10m -o CUDA_TEST ./a.out
// Get job status via
// sqjobs
// state = R means it is running
// state = D means it is done

#include <stdio.h>
#include <stdlib.h>

// OpenCL includes
#include <CL/cl.h>

const int BLOCK_SIZE_X = 18;  //interior + 2 boundary layers, ie. 16x16 + boundary layers=18x18
const int BLOCK_SIZE_Y = 18;

// OpenCL kernel
const char* programSource =
"__kernel void JacobiRelaxationGPU(__global float *u_d, __global float *f_d, int ArraySizeX, int ArraySizeY, float h)      \n"
"{                                                                                                       \n"
"  int blk_size_x=get_local_size(0);                                                                     \n"
"  int blk_size_y=get_local_size(1);                                                                     \n"
"  int tx=get_local_id(0);                                                                               \n"
"  int ty=get_local_id(1);                                                                               \n"
"  int bx=get_group_id(0)*(blk_size_x-2);                                                                \n"
"  int by=get_group_id(1)*(blk_size_y-2);                                                                \n"
"  int x=tx+bx;                                                                                          \n"
"  int y=ty+by;                                                                                          \n"
"                                                                                                        \n"
"  __local float u_sh[18][18];                                                           \n"
"                                                                                                        \n"
"  u_sh[tx][ty]=u_d[x+y*ArraySizeX];                                                                     \n"
"                                                                                                        \n"
"  barrier(CLK_LOCAL_MEM_FENCE);                                                                         \n"
"                                                                                                        \n"
"  // Interior threads only do actual work below.  Note that the 0.25f is not a typo, it means           \n"
"  // that this constant should be interpreted as a float, not a double (the default in c/c++)           \n"
"  if(tx>0 && tx < blk_size_x-1 && ty>0 && ty < blk_size_y-1) {                                          \n"
"    u_d[x+y*ArraySizeX] = 0.25f*(u_sh[tx+1][ty]+u_sh[tx-1][ty]+u_sh[tx][ty+1]+u_sh[tx][ty-1]-h*h*f_d[x+y*ArraySizeX]);                                                                                               \n"
"  }                                                                                                     \n"
"}                                                                                                       \n"
;

void chk(cl_int status, const char* cmd) {

   if(status != CL_SUCCESS) {
      printf("%s failed (%d)\n", cmd, status);
      exit(-1);
   }
}


int main(void)
{
  float *u_h, *f_h;    // pointers to host memory
  float hx=1.0;        // spatial step size
  int ArraySizeX=1026;  // Note these, minus boundary layer, have to be exactly divisible by the (BLOCK_SIZE-2) here
  int ArraySizeY=1026;
  size_t size=ArraySizeX*ArraySizeY*sizeof(float);
  FILE *fp;

  //Allocate arrays on host and initialize to zero
  u_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  f_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));

  //Initialize arrays u_h and f_h boundaries
    /* a particularly boring set of boundary conditions */
    //for(int i=0; i < ArraySizeY; i++)
    //  u_h[i*ArraySizeX] = u_h[i*ArraySizeX+ArraySizeX-1] =1.0;
    //for(int j=1; j< ArraySizeX; j++) 
    //  u_h[j] = u_h[(ArraySizeY-1)*ArraySizeX+j]=1.0;
    int i;
    for(i=ArraySizeY/4; i<3*ArraySizeY/4; i++) {
      f_h[ArraySizeY/4*ArraySizeX+i]=0.25;
      f_h[ArraySizeY/4*3*ArraySizeX+i]=-0.25;
    }

  // Use this to check the output of each API call
  cl_int status;  
     
  // Retrieve the number of platforms
  cl_uint numPlatforms = 0;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  chk(status, "clGetPlatformIDs0");
 
  // Allocate enough space for each platform
  cl_platform_id *platforms = NULL;
  platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
 
  // Fill in the platforms
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  chk(status, "clGetPlatformIDs1");

  // Retrieve the number of devices
  cl_uint numDevices = 0;
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
  chk(status, "clGetDeviceIDs0");

  // Allocate enough space for each device
  cl_device_id *devices;
  devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

  // Fill in the devices 
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
  chk(status, "clGetDeviceIDs1");
  
  // Create a context and associate it with the devices
  cl_context context;
  context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
  chk(status, "clCreateContext");

  // Create a command queue and associate it with the device 
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
  chk(status, "clCreateCommandQueue");

  // Create a buffer objects on device
  cl_mem u_d, f_d;
  u_d = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &status);
  chk(status, "clCreateBuffer");
  f_d = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
  chk(status, "clCreateBuffer");

  //Perform computation on GPU
  // Part 1 of 4: Copy data from host to device
  status = clEnqueueWriteBuffer(cmdQueue, u_d, CL_FALSE, 0, size, u_h, 0, NULL, NULL);
  chk(status, "clEnqueueWriteBuffer");
  status = clEnqueueWriteBuffer(cmdQueue, f_d, CL_FALSE, 0, size, f_h, 0, NULL, NULL);
  chk(status, "clEnqueueWriteBuffer");

  // Create a program with source code
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
  chk(status, "clCreateProgramWithSource");

  // Build (compile) the program for the device
  status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

   if(status != CL_SUCCESS) {
      printf("clBuildProgram failed (%d)\n", status);

      // Determine the size of the log
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

      // Allocate memory for the log
      char *log = (char *) malloc(log_size);

      // Get the log
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

      // Print the log
      printf("%s\n", log); 

      exit(-1);
   }

  // Create the Jacobi Iteration kernel
  cl_kernel kernel;
  kernel = clCreateKernel(program, "JacobiRelaxationGPU", &status);
  chk(status, "clCreateKernel");

  // Associate the input and output buffers with the kernel 
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &u_d);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &f_d);
  status |= clSetKernelArg(kernel, 2, sizeof(int), &ArraySizeX);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &ArraySizeY);
  status |= clSetKernelArg(kernel, 4, sizeof(float), &hx);
  chk(status, "clSetKernelArg");

  // Part 2 of 4:  Set the work item dimensions

  // Define an index space (global work size) of work 
  // items for execution. A workgroup size (local work size) is equivalent to CUDA block.
  // In OpenCL we go from local to global in order to get something similar to CUDA block/grid
  // structure.  We could have just defined the globalWorkSize and left it at that, but that
  // would rely on the system to come up with a reasonable blocking of threads. 
  size_t localWorkSize[2]={BLOCK_SIZE_X,BLOCK_SIZE_Y};
  int nBlocksX=(ArraySizeX-2)/(BLOCK_SIZE_X-2);
  int nBlocksY=(ArraySizeY-2)/(BLOCK_SIZE_Y-2);
  // Global thread index has not direct equivalent in CUDA
  size_t globalWorkSize[2] = {nBlocksX*BLOCK_SIZE_X,nBlocksY*BLOCK_SIZE_Y};

  // Execute the kernel once and check it works before going into loop.
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  chk(status, "clEnqueueNDRange");

  int nsteps;
  for (nsteps=1; nsteps < 100000; nsteps++) {
    // Part 3 of 4: Call kernel with execution configuration
    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  }   


  // Part 4 of 4: Retrieve result from device and store in u_h
  status = clEnqueueReadBuffer(cmdQueue, u_d, CL_TRUE, 0, size, u_h, 0, NULL, NULL);
  chk(status, "clEnqueueReadBuffer");

  // Output results
  fp = fopen("Solution.txt","wt");
  for (i=0; i<ArraySizeX; i++) {
    int j;
    for (j=0; j<ArraySizeY; j++)
      fprintf(fp," %f",u_h[j*ArraySizeX+i]);
    fprintf(fp,"\n");
  }
  fclose(fp);

  //Cleanup
  // Free OpenCL resources
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmdQueue);
  clReleaseMemObject(u_d);
  clReleaseMemObject(f_d);
  clReleaseContext(context);

  // Free host resources
  free(u_h);
  free(f_h);
  free(platforms);
  free(devices);

  return 0;
}
