#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
/* See lecture notes for comments */
main(int argc, char** argv)
{
   int noprocs, nid, i, n, size;
   float *a, *b, sum = 0.0, Gsum;
   FILE *fp;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);

   if(nid == 0){
      fp = fopen("DotData.txt","rt");
      fscanf(fp,"%d",&n);
      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
      if(n % noprocs){
        printf("Number of processes is not a multiple of n.\n");
        MPI_Abort(MPI_COMM_WORLD,-1);
      }
      a = (float *) calloc(n,sizeof(float));
      b = (float *) calloc(n,sizeof(float));
      for(i = 0; i < n; i++)
         fscanf(fp,"%f %f",&a[i],&b[i]);
      fclose(fp);
      size = n / noprocs;
      for(i = 1; i < noprocs; i++){
         MPI_Send(&a[size*i],size,MPI_FLOAT,i,10,MPI_COMM_WORLD);
         MPI_Send(&b[size*i],size,MPI_FLOAT,i,20,MPI_COMM_WORLD);
      }
   }
   else{
      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
      if(n % noprocs){
         printf("Number of processes is not a multiple of n.\n");
         MPI_Abort(MPI_COMM_WORLD,-1);
      }
      size =  n / noprocs;
      a = (float *) calloc(size,sizeof(float));
      b = (float *) calloc(size,sizeof(float));
      MPI_Recv(&a[0],size,MPI_FLOAT,0,10,MPI_COMM_WORLD,&status);
      MPI_Recv(&b[0],size,MPI_FLOAT,0,20,MPI_COMM_WORLD,&status);
   }

   for(i = 0; i < size; i++)
      sum += a[i] * b[i];

   MPI_Reduce(&sum,&Gsum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

   if(nid == 0)
      printf("The inner product is %f \n",Gsum);
   MPI_Finalize();
}
