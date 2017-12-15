#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
main(int argc, char** argv)
{
   int noprocs, nid, i, n, size, k;
   float *f, sum = 0.0, Gsum;
   FILE *fp;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);

   if(nid == 0){
     
	 printf("Give Number of n.\n");
      scanf("%d\n", &n);
      f = (float *) calloc(n,sizeof(float));
      for (k =0 ;k<n ;k++) f[k] = pow(2, k); 
      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
      if(n % noprocs){
        printf("Number of processes is not a multiple of n.\n");
        MPI_Abort(MPI_COMM_WORLD,-1);
      }
       printf("Loc 1\n");
            size = n / noprocs;
      for(i = 1; i < noprocs; i++){
         MPI_Send(&f[size*i],size,MPI_FLOAT,i,10,MPI_COMM_WORLD);
        		printf("Loc 2\n");
      }
   }
   else{
      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
printf("Loc 3\n");
      if(n % noprocs){
         printf("Number of processes is not a multiple of n.\n");
         MPI_Abort(MPI_COMM_WORLD,-1);
      }
printf("Loc 4\n");
         size = n/noprocs;
         f = (float *) calloc(n,sizeof(float));
      MPI_Recv(&f[0],size,MPI_FLOAT,0,10,MPI_COMM_WORLD,&status);
   }
printf("Loc 5\n");
   for(i = 0; i < size; i++)
      sum += pow(f[i], 4);
	printf("The sum for proc %d is %f \n",nid, sum);
	
   MPI_Reduce(&sum,&Gsum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	sum =0 ;
   if(nid == 0)
      printf("The total sum is %f \n",Gsum);
   MPI_Finalize();
}
