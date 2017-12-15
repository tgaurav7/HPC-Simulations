//Identity Matrix vector Multiplication
#include <stdio.h>
#include "mpi.h"
#include <math.h>
#define e 2.718
int main(int argc, char** argv)
{
int nprocs, myid, nrows,crow, ncols , rowsent,i, j, anstype, sender, k, N=6, master = 0 ; 
double **A, *b, *c;
double ans=0.0, *buff;
double begintime1, begintime2, endtime1, endtime2; 
//printf("Give the size fo Identitiy Matrix\n");
MPI_Status status;
//MPI_Datatype anstype;
// initialize MPI, determine/distribute size of arrays here
// assume A will have rows 0,nrows-1 and columns 0,ncols-1, b is 0,ncols-1
// so c must be 0,nrows-1

nrows = N;//for identity matrix
ncols = N;//for identity matrix
MPI_Init(&argc, &argv); //Initialize the MPI environment
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);//get the number of processes
MPI_Comm_rank(MPI_COMM_WORLD, &myid); //get the rank of the processes
begintime1 = MPI_Wtime();
b = (double *) calloc(ncols, sizeof(double));
for(k =0;k<ncols;k++){ 
b[k] = (double)pow(e,k); 
}
c = (double *) calloc(ncols, sizeof(double));
buff = (double *) calloc(ncols, sizeof(double));
// Master part
if (myid == master ) {
// Initialize or read in matrix A and vector b here
A = (double **) calloc(nrows, sizeof(double *));
for(k =0;k<nrows;k++){
A[k] = (double *) calloc(ncols, sizeof(double));
}
printf("A\n");
for (i=0;i<nrows;i++){
  for(j=0;j<ncols;j++){
   // if(i==j) {
      A[i][j] = i+j;
   // }
    printf("%f\t", A[i][j]);//initialising identity matrix
  }
  printf("\n");
}


// send b to every slave process, note b is an array and b=&b[0] 
MPI_Bcast(b, ncols, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
// send one row to each slave tagged with row number, assume nprocs<nrows
rowsent=0;
for (i=1; i< nprocs; i++) {
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
   MPI_Send(A[rowsent],ncols,MPI_DOUBLE_PRECISION,sender,rowsent+1,MPI_COMM_WORLD);
   rowsent++;
 }
 else        // tell sender no more work to do via a 0 TAG
   MPI_Send(MPI_BOTTOM,0,MPI_DOUBLE_PRECISION,sender,0,MPI_COMM_WORLD);
}
}
// Slave part
else {
// slaves recieve b, then compute dot products until done message recieved
MPI_Bcast(b, ncols, MPI_DOUBLE_PRECISION, master, MPI_COMM_WORLD);
MPI_Recv(buff,ncols,MPI_DOUBLE_PRECISION,master,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

while(status.MPI_TAG != 0) {
 crow = status.MPI_TAG;
 ans=0.0;
 for (i=0; i<ncols; i++)
    ans+=buff[i]*b[i];
 MPI_Send(&ans,1,MPI_DOUBLE_PRECISION, master, crow, MPI_COMM_WORLD);
 MPI_Recv(buff,ncols,MPI_DOUBLE_PRECISION,master, MPI_ANY_TAG,
 	MPI_COMM_WORLD, &status); 
}
}
// output c here on master node
if((myid==master)&&(rowsent>=nrows)){

for(k=0;k<nrows;k++){
printf("b %f\t", b[k]);
printf("C %f\n", c[k]);
}
}
endtime1 = MPI_Wtime();
printf("That took %f seconds for processor %d\n",endtime1-begintime1, myid);
MPI_Finalize();
//free any allocated space here
return 0;
}
