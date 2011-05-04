/* 11 Nov 2009
 * Matt Anderson
 * Deal with hpx output
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int floatcmp(double a,double b) {
  double epsilon = 1.e-3;
  if ( a < b + epsilon && a > b - epsilon ) return 1;
  else return 0;
}

int main(int argc, char *argv[]) {

  int i,j,gf3_rc;
  int size;
  int shape[3];
  if ( argc < 3 ) {
    printf(" Usage: debug <logcode1.dat> <logcode2.dat>\n");
    exit(0);
  }
  char basename[80];
  char basename2[80];
  char cnames[80];
  sprintf(cnames,"x");
  sprintf(basename,argv[1]);
  sprintf(basename2,argv[2]);

  FILE *fdata;
  char j1[64], j2[64], j3[64], j4[64];

  /* Ensure that the file exists */
  fdata = fopen(basename,"r");
  size = 0;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      size++;
    }
  } else {
    printf(" data file %s doesn't exist.  Try again\n",basename);
    exit(0);
  }

  /* Ensure that the file exists */
  fdata = fopen(basename2,"r");
  int size2 = 0;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      size2++;
    }
  } else {
    printf(" data file %s doesn't exist.  Try again\n",basename2);
    exit(0);
  }

  //malloc some memory
  int *level;
  int *level2;
  double *timesteps,*x,*value;
  double *timesteps2,*x2,*value2;
  level = (int *) malloc(sizeof(int)*size);
  timesteps = (double *) malloc(sizeof(double)*size);
  x = (double *) malloc(sizeof(double)*size);
  value = (double *) malloc(sizeof(double)*size);

  level2 = (int *) malloc(sizeof(int)*size2);
  timesteps2 = (double *) malloc(sizeof(double)*size2);
  x2 = (double *) malloc(sizeof(double)*size2);
  value2 = (double *) malloc(sizeof(double)*size2);

  fdata = fopen(basename,"r");
  i = 0;
  int maxlevel = -1;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      level[i] = atoi(j1);
      if (level[i] > maxlevel) maxlevel = level[i];

      timesteps[i] = atof(j2);
      x[i] = atof(j3);
      value[i] = atof(j4);
      i++;
    }
  } else {
    printf(" PROBLEM\n");
    exit(0);
  }
  maxlevel++;

  fdata = fopen(basename2,"r");
  j = 0;
  int maxlevel2 = -1;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      level2[j] = atoi(j1);
      if (level2[j] > maxlevel2) maxlevel2 = level2[j];

      timesteps2[j] = atof(j2);
      x2[j] = atof(j3);
      value2[j] = atof(j4);
      j++;
    }
  } else {
    printf(" PROBLEM\n");
    exit(0);
  }
  maxlevel2++;

  // Compare what's in list 1 with list 2 -- find any coincident entries; compare on x
  int count = 0;
  int k,l;
  double tmpx,tmpx2;
  for (k=0;k<i;k++) {
    tmpx = x[k];
    for (l=0;l<j;l++) {
      tmpx2 = x2[l];
      if ( floatcmp(tmpx,tmpx2) == 1 ) {
        // compare the timestep
        if ( floatcmp(timesteps[k],timesteps2[l]) == 1 ) {
          // found a coincident
          printf(" x: %g timestep: %g x2: %g timestep: %g\n",x[k],timesteps[k],x2[l],timesteps2[l]);
          count++;
        }
      }
    }
  }
  printf("\n\n Coincidences: %d\n",count);

  return 0;
}
