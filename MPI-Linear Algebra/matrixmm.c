//Inverse Matrix Multiplication
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#define e 2.718
int main(int argc, char** argv)
{
   int nprocs, anstype, myid, rowsent,  sender, Arows=6, Acols=6, Brows, Bcols=6, Crows, Ccols, crow , i, j, k, N, master = 0 ; 
	double **A, **B, **C;
	double *ans, *buff;
   MPI_Status status;
	// initialize MPI, determine/distribute size of arrays here
	// assume A will have rows 0,nrows-1 and columns 0,ncols-1, b is 0,ncols-1
	// so c must be 0,nrows-1
   //printf("Give the size A and B Matrix as Rows A, Cols A, Cols B"); // size in for matrix
   //scanf("%d\t%d\t%d\n", Arows, Acols, Bcols);
   Brows = Acols ;
   Crows = Arows;
   Ccols = Bcols;
   MPI_Init(&argc, &argv); //Initialize the MPI environment
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);//get the number of processes
   MPI_Comm_rank(MPI_COMM_WORLD, &myid); //get the rank of the processes
    B = (double **) calloc(Brows, sizeof(double *) );
	 for(k =0;k<Brows;k++){
		B[k] = (double *) calloc( Bcols, sizeof(double) );
	 }
	 C = (double **) calloc(Crows, sizeof(double *) );
	 for(k =0;k<Crows;k++){
		C[k] = (double *) calloc( Ccols, sizeof(double));
	 }
	 printf("2\n");
	ans = (double *) calloc(Ccols, sizeof(double));
    buff = (double *) calloc(Acols,  sizeof(double));
    printf("a %d\n", myid);
  // Master part
 if (myid == master ) {
   // Initialize or read in matrix A and vector b here
	 A = (double **) calloc(Arows, sizeof(double *) );
	 for(k =0;k<Arows;k++){
		A[k] = (double *) calloc(Acols, sizeof(double) );
	 }
	
		for (i=0;i<Arows;i++){
			for(j=0;j<Acols;j++){
				A[i][j] = i+j; 
			}
		}
	 for (i=0;i<Bcols;i++){
		for(j=0;j<Bcols;j++){
			B[i][j] = i+j;
			}
	}
	 
	// send b to every slave process, note b is an array and b=&b[0] 
	MPI_Bcast(&(B[0][0]),Bcols*Brows, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
 
   // send one row to each slave tagged with row number, assume nprocs<nrows
   rowsent=0;
   for (i=1; i<nprocs; i++) {
	   printf(" myid3 %d\n",i);
     // Note A is a 2D array so A[rowsent]=&A[rowsent][0]
     MPI_Send(A[rowsent], Acols, MPI_DOUBLE_PRECISION,i,rowsent+1,MPI_COMM_WORLD);
     rowsent++;
   }
     printf("%d 5\n", Arows);
   for (i=0; i<Arows; i++) {
    MPI_Recv(ans, 1, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, MPI_ANY_TAG,
	       MPI_COMM_WORLD, &status);
      
     sender = status.MPI_SOURCE;
     anstype = status.MPI_TAG;            //row number+1
	 for(i=0;i<Arows;i++){
     C[anstype-1][i] = ans[i];
	 }
	 for(i=0;i<Arows;i++){printf("C anstype-1 %f ", C[anstype-1][i]);}
     printf("%f %d %d 6\n", ans[1], anstype, sender);
     if (rowsent < Arows) {                // send new row
       MPI_Send(A[rowsent],Acols,MPI_DOUBLE_PRECISION,sender,rowsent+1,MPI_COMM_WORLD);
       rowsent++;
     }
     else        // tell sender no more work to do via a 0 TAG
       MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE_PRECISION,sender,0,MPI_COMM_WORLD);
   }
 }

 // Slave part
 else {
 	printf("ELSE\n");
   // slaves recieve b, then compute dot products until done message recieved
	MPI_Bcast(&(B[0][0]),Bcols*Brows, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
   for(i=0;i<Brows;i++){
	   for(j=0;j<Bcols;j++){
		   printf("b %d here %f\t", myid, B[i][j]);
		   }
		   printf("\n");
		}
    MPI_Recv(buff, Acols, MPI_DOUBLE_PRECISION,master, MPI_ANY_TAG,
	       MPI_COMM_WORLD, &status);
	for(i=0;i<Acols;i++){
		printf("buff %d here %f\n", myid, buff[i]);
		}
		   while(status.MPI_TAG != 0){
			   crow = status.MPI_TAG;
			   for(i=0;i<Bcols;i++){
				  ans[i] = 0.0;
				  for(j=0;j<Acols;j++){
					  ans[i] += buff[j]*B[j][i];
				  
					printf("ans3 %f\n", ans[i]);
				}
			  }
			  MPI_Send(ans, Bcols, MPI_DOUBLE_PRECISION, master, crow, MPI_COMM_WORLD);
			   printf("9 let's see %d\n", myid);
	MPI_Recv(buff, Acols, MPI_DOUBLE_PRECISION, master, MPI_ANY_TAG, 
			  MPI_COMM_WORLD, &status);
	}
 }
// output c here on master node
for(i=0;i<Crows;i++){
	for(j=0;j<Crows;j++){
		printf("%f\t", C[i][j]);
	}
	printf("\n");
}
 MPI_Finalize();
 //free any allocated space here
  return 0;
}

