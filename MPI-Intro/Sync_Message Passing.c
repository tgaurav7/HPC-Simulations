#include <stdio.h>
#include "mpi.h"

main(int argc, char** argv)
{
   int i, N, j, noprocs, task, taskx, nid, node;
   float  sum = 0, Gsum, m=1000000, l=1000000;
   MPI_Status status;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);

for(task=1;task<=10;task++){
	node=task-noprocs*(task/noprocs);
	if(node==nid){
		if(node==2){
						volatile unsigned long long i;
  						for (i = 0; i < 1000000000ULL; ++i) m = m/2;
  			 }
  			 MPI_Send(&task, 1, MPI_INTEGER, 0, 10, MPI_COMM_WORLD);//Tell master i am the one
	}
	if(nid==0){
		MPI_Recv(&taskx, 1, MPI_INTEGER, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
		printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
	}
}

for(task=11;task<=20;task++){
	node = task-noprocs*(task/noprocs);
	if(node==nid){
	if(node==2){
						volatile unsigned long long j;
  						for (j = 0; j < 1000000000ULL; ++j) l = l/2;
  			 }
	MPI_Send(&task, 1, MPI_INTEGER, 0, 10, MPI_COMM_WORLD);
}
if(nid==0){
		MPI_Recv(&taskx, 1, MPI_INTEGER, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
		printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
	}
}
MPI_Finalize();
}
