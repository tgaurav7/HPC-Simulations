#include <stdio.h>
#include "mpi.h"
main(int argc, char** argv)
{
   int noprocs, nid;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &nid);

   printf("Hello from processor %i of %i \n",nid,noprocs);

   MPI_Finalize();
}
