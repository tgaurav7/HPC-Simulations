#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv)
{
   int noprocs, nid, N, k, i,j, sum=0;int f[100];
  	
	
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);
 if(nid == 0){
      printf("Please enter the number of terms N -> ");
      scanf("%d",&N);
	}
MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD); 
	for(i=0;i<N;i++) {f[i]=(-1)^i;}
   printf("Calculating for processor %i of %i \n",nid,noprocs);
   for(j=0;j<N;j++)  {sum += pow(f[j], nid);   }                                                                    
   
   printf("sum from processor %d is %d \n", nid, sum);
	for(i=0;i<N;i++) {f[i]=0;}

   MPI_Finalize();
   
   return 0;
}
