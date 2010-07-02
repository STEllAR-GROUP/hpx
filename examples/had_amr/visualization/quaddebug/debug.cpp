/* 11 Nov 2009
 * Matt Anderson
 * Deal with hpx output
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sstream>
#include <string>
#include "mpreal.h"



int floatcmp(mpfr::mpreal a,mpfr::mpreal b) {
  mpfr::mpreal epsilon = 1.e-17;
  if ( a < b + epsilon && a > b - epsilon ) return 1;
  else return 0;
}

int main(int argc, char *argv[]) {

  int i,j,gf3_rc;
  int size;
  int shape[3];
  if ( argc < 3 ) {
    printf(" Usage: debug <unigrid.dat> <amr.dat>\n");
    exit(0);
  }

  mpfr::mpreal::set_default_prec(128);

  char basename[80];
  char basename2[80];
  char cnames[80];
  sprintf(cnames,"x");
  sprintf(basename,argv[1]);
  sprintf(basename2,argv[2]);

  FILE *fdata;
  char j1[164], j2[164], j3[164], j4[164];

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
  mpfr::mpreal *timesteps,*x,*value;
  mpfr::mpreal *timesteps2,*x2,*value2;
  level = (int *) malloc(sizeof(int)*size);
  timesteps = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size);
  x = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size);
  value = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size);

  level2 = (int *) malloc(sizeof(int)*size2);
  timesteps2 = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size2);
  x2 = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size2);
  value2 = (mpfr::mpreal *) malloc(sizeof(mpfr::mpreal)*size2);

  fdata = fopen(basename,"r");
  i = 0;
  int maxlevel = -1;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      level[i] = atoi(j1);
      if (level[i] > maxlevel) maxlevel = level[i];

      std::string s = str(j2);
      timesteps[i] = s.c_str();
      std::string s = str(j3);
      x[i] = s.c_str();
      std::string s = str(j4);
      value[i] = s.c_str();
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

      std::string s = str(j2);
      timesteps2[j] = s.c_str();
      std::string s = str(j3);
      x2[j] = s.c_str();
      std::string s = str(j4);
      value2[j] = s.c_str();
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
  mpfr::mpreal tmpx,tmpx2;
  for (k=0;k<i;k++) {
    tmpx = x[k];
    for (l=0;l<j;l++) {
      tmpx2 = x2[l];
      if ( floatcmp(tmpx,tmpx2) == 1 ) {
        // compare the timestep
        if ( floatcmp(timesteps[k],timesteps2[l]) == 1 ) {
          // compare the value
          if ( floatcmp(value[k],value2[l]) != 1 ) {
            std::cout << " x " << x[k] << " time " << timesteps[k] << " unigrid " << value[k] << " amr " << value2[l] << std::endl;
          }
        }
      }
    }
  }

  return 0;
}
