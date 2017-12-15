/* Compile line:
gcc -O3 random_number.c -lm  -o ranj
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

double gasdev(int ), unidev(int );

int main(int argc, char** argv) 
{
  int seed;

  /* initialize random number generator */
  seed=45156557;
  srandom(seed);

  printf("A gaussian random number %f\n",gasdev(0)); 
  printf("A uniform random number %f\n",(double)random()/(double)RAND_MAX); 

}

double gasdev(int start)
{
  /***************************************************************************/
  /* Generates random numbers from a Gaussian distribution. Based on routine */
  /* from Numerical Recipies.                                                */
  /***************************************************************************/
  static int iset=0;
  static double gset;
  double fac,rsq,v1,v2;

  if (start==1) {
    iset=0;
  }
  if  (iset == 0) {
    do {
      v1=2.0*(double)random()/(double)RAND_MAX-1.0;
      v2=2.0*(double)random()/(double)RAND_MAX-1.0;
      rsq=v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac=sqrt(-2.0*log(rsq)/rsq);
    gset=v1*fac;
    iset=1;
    return v2*fac;
  } else {
    iset=0;
    return gset;
  }
}

