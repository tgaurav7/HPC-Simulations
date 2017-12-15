#include <stdio.h>
#include "mpi.h"
/* See lecture notes for comments */
main(int argc, char** argv)
{
   int i, N, noprocs, nid;
   float  sum = 0, Gsum;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);

   if(nid == 0){
      printf("Please enter the number of terms N -> ");
      scanf("%d",&N);
   }
   MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
   for(i = nid; i < N; i += noprocs)
      if(i % 2)
         sum -= (float) 1 / (i + 1);
      else
         sum += (float) 1 / (i + 1);
   MPI_Reduce(&sum,&Gsum,1,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

   if(nid == 0)
      printf("An estimate of ln(2) is %f \n",Gsum);
   MPI_Finalize();
}
