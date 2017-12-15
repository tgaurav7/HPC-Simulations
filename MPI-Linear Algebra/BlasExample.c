#include <stdio.h>
#include <gsl/gsl_blas.h>
// This example is taken from the gsl manual.  The original can be found at:
// http://www.gnu.org/software/gsl/manual/gsl-ref_13.html#SEC234
// I compiled with: gcc -O3 BlasExample.c -lgsl
// This appears to work on most LINUX/CYGWIN systems, if you have trouble
// try looking at:
// http://www.gnu.org/software/gsl/manual/gsl-ref_2.html
// for some other tips.
int
main (void)
{
  int i;
  double a[] = { 0.11, 0.12, 0.13,
                 0.21, 0.22, 0.23 };

  double b[] = { 1011, 1012,
                 1021, 1022,
                 1031, 1032 };

  double c[] = { 1.00, 2.00,
                 3.00, 4.00 };
for(i=0;i<4;i++)
  printf("%g\n\n", c[i]);

  gsl_matrix_view A = gsl_matrix_view_array(a, 2, 3);
  gsl_matrix_view B = gsl_matrix_view_array(b, 3, 2);
  gsl_matrix_view C = gsl_matrix_view_array(c, 2, 2);
for(i=0;i<4;i++)
  printf("%g", c[i]);

  /* Compute C = A B */

  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                  1.0, &A.matrix, &B.matrix,
                  0.0, &C.matrix);

  printf ("[ %g, %g\n", c[0], c[1]);
  printf ("  %g, %g ]\n", c[2], c[3]);

  return 0;  
}
