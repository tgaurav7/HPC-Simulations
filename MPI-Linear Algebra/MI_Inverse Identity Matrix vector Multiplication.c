//Inverse Identity Matrix vector Multiplication
#include <stdio.h>
#include "mpi.h"
#include <math.h>
int main(int argc, char** argv)
{
   int noprocs, nid, anstype, sender, rowsent, nrows, ncols , i, j, k, N, master = 0 ; 
	double **A, *b, *c;
	double ans, buff;
   MPI_Status status;
   MPI_Datatype anstype;
	// initialize MPI, determine/distribute size of arrays here
	// assume A will have rows 0,nrows-1 and columns 0,ncols-1, b is 0,ncols-1
	// so c must be 0,nrows-1
   printf("Give the size fo Identitiy Matrix"); // size in for matrix
   scanf("%d\n", N);
   nrows = N;//for identity matrix
   ncols = N;//for identity matrix
   MPI_Init(&argc, &argv); //Initialize the MPI environment
   MPI_Comm_size(MPI_COMM_WORLD, &noprocs);//get the number of processes
   MPI_Comm_rank(MPI_COMM_WORLD, &nid); //get the rank of the processes
   
  // Master part
 if (myid == master ) {
   // Initialize or read in matrix A and vector b here
	 A = (double **)calloc(sizeof(double *) * nrows);
	 for(k =0;k<nrows;k++){
		A[k] = (double *) calloc(sizeof(double) * ncols);
	 }
		for (i=0;i<nrows;i++){
			for(y=0;y<ncols;y++){
				A[i][j] = i*j;//initialising the matrix 
			}
		}
	b = (double *) calloc(sizeof(double) * ncols);
	 for(k =0;k<ncols;k++){	b[k] = (double)pow(e,k); }
	c = (double *) calloc(sizeof(double) * nrows);
	
	// send b to every slave process, note b is an array and b=&b[0] 
	MPI_Bcast(b,ncols, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
 
   // send one row to each slave tagged with row number, assume nprocs<nrows
   rowsent=0;
   for (i=1; i< numprocs-1; i++) {
     // Note A is a 2D array so A[rowsent]=&A[rowsent][0]
     MPI_Send(A[rowsent], ncols, MPI_DOUBLE_PRECISION,i,rowsent+1,MPI_COMM_WORLD);
     rowsent++;
   }
   
   
      for (i=0; i<nrows; i++) {
     MPI_Recv(&ans, 1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, MPI_ANY_TAG,
	       MPI_COMM_WORLD, &status);
     sender = status.MPI_SOURCE;
     anstype = status.MPI_TAG;            //row number+1
     c[anstype-1] = ans;
     if (rowsent < nrows) {                // send new row
       MPI_Send(A[rowsent+1],ncols,MPI_DOUBLE_PRECISION,sender,rowsent+1,MPI_COMM_WORLD);
       rowsent++;
     }
     else        // tell sender no more work to do via a 0 TAG
       MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE_PRECISION,sender,0,MPI_COMM_WORLD);
   }
 }

 // Slave part
 else {
   // slaves recieve b, then compute dot products until done message recieved
   MPI_Bcast(b,ncols, MPI_DOUBLE_PRECSION, master, MPI_COMM_WORLD);
   
   MPI_Recv(buff,ncols,MPI_DOUBLE_PRECISION,master,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   while(status.MPI_TAG != 0) {
     crow = status.MPI_TAG;
     ans=0.0;
     for (i=0; i< ncols; i++)
       ans+=buff[i]*b[i];
     MPI_Send(&ans,1,MPI_DOUBLE_PRECISION, master, crow, MPI_COMM_WORLD);
     MPI_Recv(buff,ncols,MPI_DOUBLE_PRECISION,master,MPI_ANY_TAG,MPI_COMM_WORLD,&status); 
   }
 }
// output c here on master node
for(k=0;k<nrows;k++){
	printf("%f\n", c[k]);
}
 MPI_FINALIZE();
 //free any allocated space here
  return 0;
}