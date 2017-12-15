//Inverse Matrix Multiplication
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#define e 2.718
int main(int argc, char** argv)
{
int nprocs, anstype, myid, rowsent,  sender, Arows=6, Acols=6, Brows, Bcols=6, Crows, Ccols, crow , i, j, k, N, master = 0 ; 
double **A, B[6][6], C[6][6];
double ans[6]; 
double buff[6];
double begintime1, begintime2, endtime1, endtime2; 
MPI_Status status;
// initialize MPI, determine/distribute size of arrays here
// assume A will have rows 0,nrows-1 and columns 0,ncols-1, b is 0,ncols-1
// so c must be 0,nrows-1

Brows = Acols ;
Crows = Arows;
Ccols = Bcols;
MPI_Init(&argc, &argv); //Initialize the MPI environment
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);//get the number of processes
MPI_Comm_rank(MPI_COMM_WORLD, &myid); //get the rank of the processes
begintime1 = MPI_Wtime();
// Master part
printf("Rank %d\n", &myid)
if (myid == master ) {
// Initialize or read in matrix A and vector b here
 A = (double **) calloc(Arows, sizeof(double *) );
 for(k =0;k<Arows;k++){
	A[k] = (double *) calloc(Acols, sizeof(double) );
 }
printf("A\n");
	for (i=0;i<Arows;i++){
		for(j=0;j<Acols;j++){
			A[i][j] = i+j; 
			printf("%f\t", A[i][j]);
		}
		printf("\n");
	}
	printf("B\n");
 for (i=0;i<Bcols;i++){
	for(j=0;j<Bcols;j++){
		B[i][j] = i+j;
		printf("%f\t", B[i][j]);
		}
		printf("\n");
}
 
// send b to every slave process, note b is an array and b=&b[0] 
MPI_Bcast(B,Bcols*Brows, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);

// send one row to each slave tagged with row number, assume nprocs<nrows
rowsent=0;
for (i=1; i<nprocs; i++) {
 // Note A is a 2D array so A[rowsent]=&A[rowsent][0]
 MPI_Send(A[rowsent], Acols, MPI_DOUBLE_PRECISION,i,rowsent+1,MPI_COMM_WORLD);
 rowsent++;
}
for (i=0; i<Arows; i++) {
MPI_Recv(ans, Crows, MPI_DOUBLE_PRECISION, MPI_ANY_SOURCE, MPI_ANY_TAG,
       MPI_COMM_WORLD, &status);
  
 sender = status.MPI_SOURCE;
 anstype = status.MPI_TAG;            //row number+1
 for(i=0;i<Arows;i++){
 C[anstype-1][i] = ans[i];
 }
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
// slaves recieve b, then compute dot products until done message recieved
MPI_Bcast(B,Bcols*Brows, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
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
for(j=0;j<Ccols;j++){
	printf("%f\t", C[i][j]);
}
printf("\n");
}
endtime1 = MPI_Wtime();
printf("That took %f seconds for processor %d\n",endtime1-begintime1, myid);
MPI_Finalize();
//free any allocated space here
return 0;
}

