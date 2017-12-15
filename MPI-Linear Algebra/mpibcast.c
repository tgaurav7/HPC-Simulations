#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
        int rank;
        int buf;
        const int root=0;
	int a[2][4] = {{1, 2, 3, 4}, {23,4,5,6}};
	int i;
	int b[4] = {0, 0, 0, 0};
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(rank == root) {
           buf = 777;
        }

        printf("[%d]: Before Bcast, buf is %d\n", rank, buf);

        /* everyone calls bcast, data is taken from root and ends up in everyone's buf */
        MPI_Bcast(&buf, 1, MPI_INT, root, MPI_COMM_WORLD);
	for(i=0;i<4;i++)
	b[i] = a[0][i];
	for(i =0;i<4;i++){printf("%d\n", b[i]);}
        printf("[%d]: After Bcast, buf is %d\n", rank, buf);

        MPI_Finalize();
        return 0;
}