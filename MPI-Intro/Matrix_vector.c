////////////////////////////////////////////
// Matrix-vector multiplication code Ab=c //
////////////////////////////////////////////

// Note that I will index arrays from 0 to n-1.
// Here slaves do all the work and master just handles collating results
// and sending infor about A.

// include, definitions, globals etc here
{
 MPI_Status status;

 // initialize MPI, determine/distribute size of arrays here
 // assume A will have rows 0,nrows-1 and columns 0,ncols-1, b is 0,ncols-1
 // so c must be 0,nrows-1

 // Master part
 if (myid == master ) {
   // Initialize or read in matrix A and vector b here

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

 MPI_FINALIZE();
 //free any allocated space here
}




///////////////////////////////////////////////////
// changes for Matrix-Matrix multiplication code //
///////////////////////////////////////////////////

{

  // changes to master part: you figure it out

  //changes to slave part...
    // slaves recieve B, then compute rows of C until done message recieved
    MPI_Bcast(B,Bcols*Brows, MPI_DOUBLE_PRECSION, master, MPI_COMM_WORLD);

    MPI_Recv(buff, Acols, MPI_DOUBLE_PRECISION, master, MPI_ANY_TAG,
	       MPI_COMM_WORLD, &status);
    while (status.MPI_TAG != 0) {
      crow = status.MPI_TAG;
      for (i=1; i<= Bcols) {
	ans[i]=0.0;
	for (j=1; j<= Acols; j++)
	  ans[i] += buff[j]*b[j][i];
      }
      MPI_Send(ans,Bcols,MPI_DOUBLE_PRECISION, master, crow, MPI_COMM_WORLD);
      MPI_Recv(buff, Acols, MPI_DOUBLE_PRECISION, master, MPI_ANY_TAG,
	       MPI_COMM_WORLD, &status);
    }

    // blah, blah, blah...
  
