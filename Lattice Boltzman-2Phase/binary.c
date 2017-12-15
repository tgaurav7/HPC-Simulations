/* Standard Lattice Boltzmann Binary fluid a la Alex thesis */
/* Complie line on Ultra sun
cc -fast -xO5 binary.c -lm -o binary
*/
/* Compile line on DEC
cc -O3 binary.c -lm -o binary
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parametercalc(void);
void streamout(void);
void equilibriumdist(void);
void initialize(void);

#define L 50
#define Nmax 100000000

#define temperature 0.5
#define kappa 0.02
#define gamma 2.0
#define lambda 1.1
#define tau1 1.0
#define tau2 1.0
#define dt 1.0

#define Pi 3.141592653589793
#define TwoPi 6.283185307179586

double f1[L][L][9],g1[L][L][9],f2[L][L][9],g2[L][L][9];
double Fc[L][L][9],Gc[L][L][9];
double feq[L][L][9],geq[L][L][9];
double density[L][L],phi[L][L],u[L][L][2];
int e[9][2];
double (*f)[L][9],(*g)[L][9],(*fnew)[L][9],(*gnew)[L][9];

int main(int argc, char** argv) 
{
  int i,j,k,n;
  int iup,idwn,jup,jdwn;
  double (*tmp)[L][9];

  initialize();
  printf("%f %f %i %i %i %i\n",1.0,1.0,L,L,1,2);
  printf("vx\n");
  printf("vy\n");
  printf("density\n");
  printf("phi\n");
  
  for (n=0; n<Nmax; n++) {
    parametercalc();
    if (n==750) {
      for (i=0; i<L; i++) {
	for (j=0; j<L; j++) {
	  u[i][j][1]+=0.1;
	}
      }
    }
    streamout();
    equilibriumdist();
    
    for (i=0; i<L; i++)
      for (j=0; j<L; j++)
	for (k=0; k<9; k++) {
  	  Fc[i][j][k]= (feq[i][j][k]-f[i][j][k])/tau1; 
  	  Gc[i][j][k]= (geq[i][j][k]-g[i][j][k])/tau2; 
	}
    
    for (i=0; i<L; i++) {
      if (i==L-1) iup=0; else iup=i+1;
      if (i==0) idwn=L-1; else idwn=i-1;
      for (j=0; j<L; j++) {
	if (j==L-1) jup=0; else jup=j+1;
	if (j==0) jdwn=L-1; else jdwn=j-1;

	fnew[i][j][2]=f[i][jdwn][2]+dt*Fc[i][jdwn][2];
	fnew[i][j][0]=f[i][j][0]+dt*Fc[i][j][0];
	fnew[i][j][4]=f[i][jup][4]+dt*Fc[i][jup][4];
	fnew[i][j][5]=f[idwn][jdwn][5]+dt*Fc[idwn][jdwn][5];
	fnew[i][j][1]=f[idwn][j][1]+dt*Fc[idwn][j][1];
	fnew[i][j][8]=f[idwn][jup][8]+dt*Fc[idwn][jup][8];
	fnew[i][j][6]=f[iup][jdwn][6]+dt*Fc[iup][jdwn][6];
	fnew[i][j][3]=f[iup][j][3]+dt*Fc[iup][j][3];
	fnew[i][j][7]=f[iup][jup][7]+dt*Fc[iup][jup][7];

	gnew[i][j][2]=g[i][jdwn][2]+dt*Gc[i][jdwn][2];
	gnew[i][j][0]=g[i][j][0]+dt*Gc[i][j][0];
	gnew[i][j][4]=g[i][jup][4]+dt*Gc[i][jup][4];
	gnew[i][j][5]=g[idwn][jdwn][5]+dt*Gc[idwn][jdwn][5];
	gnew[i][j][1]=g[idwn][j][1]+dt*Gc[idwn][j][1];
	gnew[i][j][8]=g[idwn][jup][8]+dt*Gc[idwn][jup][8];
	gnew[i][j][6]=g[iup][jdwn][6]+dt*Gc[iup][jdwn][6];
	gnew[i][j][3]=g[iup][j][3]+dt*Gc[iup][j][3];
	gnew[i][j][7]=g[iup][jup][7]+dt*Gc[iup][jup][7];
      }
    }
    tmp=f;
    f=fnew;
    fnew=tmp;
    tmp=g;
    g=gnew;
    gnew=tmp;
  }
  
}


void equilibriumdist(void)
{
  double A0,A1,A2,B1,B2,C0,C1,C2,D1,D2,G1xx,G2xx,G1xy,G2xy,G1yy,G2yy;
  double H0,H1,H2,K1,K2,J0,J1,J2,Q1,Q2;
  double dphidx,dphidy,rho,phiij,mu,usq,udote,laplacian;
  int i,j,k,iup,idwn,jup,jdwn;

  for (i=0; i<L; i++) {
    if (i==L-1) iup=0; else iup=i+1;
    if (i==0) idwn=L-1; else idwn=i-1;
    for (j=0; j<L; j++) {
      if (j==L-1) jup=0; else jup=j+1;
      if (j==0) jdwn=L-1; else jdwn=j-1;
      rho=density[i][j];
      phiij=phi[i][j];
      laplacian=kappa*((phi[iup][j]-2.0*phiij+phi[idwn][j])+
		       (phi[i][jup]-2.0*phiij+phi[i][jdwn]));
      mu= -lambda/2.0*phiij/rho+
	  temperature/2.0*log(((rho+phiij)/(rho-phiij)))-laplacian;
      A2= (rho*temperature-phiij*laplacian)/8.0;
      A1= 2.0*A2;
      A0= rho-12.0*A2;
      B2= rho/12.0;
      B1= 4.0*B2;
      C2= -rho/16.0;
      C1= 2.0*C2;
      C0= -3.0*rho/4.0;
      D2= rho/8.0;
      D1= 4.0*D2;
      dphidx=(phi[iup][j]-phi[idwn][j])/2.0;
      dphidy=(phi[i][jup]-phi[i][jdwn])/2.0;
      G2xx= kappa/16.0*(dphidx*dphidx-dphidy*dphidy);
      G2xy= kappa/8.0*dphidx*dphidy;
      G2yy= -G2xx;
      G1xx= 4.0*G2xx;
      G1xy= 4.0*G2xy;
      G1yy= 4.0*G2yy;
      H2= gamma/8.0*mu;
      H1= 2.0*H2;
      H0= phiij-12.0*H2;
      K2= phiij/12.0;
      K1= 4.0*K2;
      J2= -phiij/16.0;
      J1= 2.0*J2;
      J0= -3.0*phiij/4.0;
      Q2= phiij/8.0;
      Q1= 4.0*Q2;
      
      usq=u[i][j][0]*u[i][j][0]+u[i][j][1]*u[i][j][1];
      feq[i][j][0]=A0+C0*usq;
      geq[i][j][0]=H0+J0*usq;
      for (k=1; k<=4; k++) {
	udote=u[i][j][0]*e[k][0]+u[i][j][1]*e[k][1];
	feq[i][j][k]=A1+B1*udote+C1*usq+D1*udote*udote+G1xx*e[k][0]*e[k][0]+
	  2.0*G1xy*e[k][0]*e[k][1]+G1yy*e[k][1]*e[k][1];
	geq[i][j][k]=H1+K1*udote+J1*usq+Q1*udote*udote;
      }
      for (k=5; k<=8; k++) {
	udote=u[i][j][0]*e[k][0]+u[i][j][1]*e[k][1];
	feq[i][j][k]=A2+B2*udote+C2*usq+D2*udote*udote+G2xx*e[k][0]*e[k][0]+
	  2.0*G2xy*e[k][0]*e[k][1]+G2yy*e[k][1]*e[k][1];
	geq[i][j][k]=H2+K2*udote+J2*usq+Q2*udote*udote;
      }
    }
  }
}


void streamout(void)
{
  int i,j;
  
  for(i=0; i<L; i++) {
    for (j=0; j<L; j++) {
      printf("%16.12f %16.12f %16.12f %16.12f\n",
	     u[i][j][0],u[i][j][1],density[i][j],phi[i][j]);
    }
  }
}


void parametercalc(void)
{
  int i,j,k;

  for (i=0; i<L; i++) {
    for (j=0; j<L; j++) {
      density[i][j]=0.0;
      phi[i][j]=0.0;
      u[i][j][0]=u[i][j][1]=0.0;
      for (k=0; k<9; k++) {
	density[i][j] += f[i][j][k];
	phi[i][j] += g[i][j][k];
	u[i][j][0] += f[i][j][k]*e[k][0];
	u[i][j][1] += f[i][j][k]*e[k][1];
      }
      u[i][j][0]=u[i][j][0]/density[i][j];
      u[i][j][1]=u[i][j][1]/density[i][j];
    }
  }
}


void initialize(void)
{
  int i,j,k;

  e[0][0]= 0;
  e[0][1]= 0;
  e[1][0]= 1;
  e[1][1]= 0;
  e[2][0]= 0;
  e[2][1]= 1;
  e[3][0]= -1;
  e[3][1]= 0;
  e[4][0]= 0;
  e[4][1]= -1;
  e[5][0]= 1;
  e[5][1]= 1;
  e[6][0]= -1;
  e[6][1]= 1;
  e[7][0]= -1;
  e[7][1]= -1;
  e[8][0]= 1;
  e[8][1]= -1;

  f=f1;
  g=g1;
  fnew=f2;
  gnew=g2;
  for (i=0; i<L; i++) {
    for (j=0; j<L; j++) {
      density[i][j]=2.0;
      /*phi[i][j]= 0.0+0.04*(drand48()-0.5);*/
      u[i][j][0]=0.0;
      u[i][j][1]=0.0;
/*    phi[i][j] = 1.12*(1.0+tanh((1.0*i-35.0)/4.0)-tanh((1.0*i-10.0)/4.0)); */
/*       if (i== L/2 && j==L/2) phi[i][j]=0.1; */
/*       if (i>L/2) phi[i][j]= 1.12 ; */
      if ((i-L/2)*(i-L/2)+(j-L/2)*(j-L/2) < 64)
	phi[i][j]=1.0;
      else
	phi[i][j]= -1.0;
      for (k=0; k<9; k++) {
	f[i][j][k]=density[i][j]/9.0;
	g[i][j][k]=phi[i][j]/9.0;
      }
    }
  }

  /*
  phi[L/2][L/2]=1.5;
  for (k=0; k<9; k++) {
    g[L/2][L/2][k]=phi[L/2][L/2]/9.0;
  }
  */
}




