/* 11 Nov 2009
 * Matt Anderson
 * Deal with hpx output
 */
#include <stdio.h>
#include <stdlib.h>
#include <sdf.h>
#include <math.h>


int floatcmp(double a,double b) {
  double epsilon = 1.e-6;
  if ( a < b + 1.e-6 && a > b - 1.e-6 ) return 1;
  else return 0;
}

int main(int argc, char *argv[]) {

  int i,j,k,gf3_rc;
  int size;
  int shape[3];
  if ( argc < 2 ) {
    printf(" Usage: hpx2sdf <basename>\n");
    exit(0);
  }
  char basename[80];
  char data_basename[80];
  char cnames[80];
  sprintf(cnames,"x");
  sprintf(basename,argv[1]);

  sprintf(data_basename,"%s.dat",basename);

  FILE *fdata;
  char j1[64], j2[64], j3[64], j4[64];

  /* Ensure that the file exists */
  fdata = fopen(data_basename,"r");
  size = 0;
  if ( fdata ) {
    while(fscanf(fdata,"%s %s %s %s",&j1,&j2,&j3,&j4) > 0 ) {
      size++;
    }
  } else {
    printf(" data file %s.dat doesn't exist.  Try again\n",basename);
    exit(0);
  }

  //malloc some memory
  int *level;
  double *timesteps,*x,*value;
  level = (int *) malloc(sizeof(int)*size);
  timesteps = (double *) malloc(sizeof(double)*size);
  x = (double *) malloc(sizeof(double)*size);
  value = (double *) malloc(sizeof(double)*size);

  fdata = fopen(data_basename,"r");
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

  // Get a list of unique timesteps for binning
  int timestep_size = 1;
  int max_num_timesteps = 10000;
  double *num_timesteps;
  int found = 0;
  num_timesteps = (double *) malloc(sizeof(double)*max_num_timesteps);
  num_timesteps[0] = timesteps[0];
  for (i=0;i<size;i++) {
    // see if timestep[i] has already been recorded
    found = 0;
    for (j=0;j<timestep_size;j++) {
      if ( floatcmp(timesteps[i],num_timesteps[j]) == 1 ) {
        found = 1;
        break;
      }
    }
    if (found == 0) {
      // timestep[i] has not been recorded in num_timesteps array, so add it now
      num_timesteps[timestep_size] = timesteps[i];
      timestep_size++;
    }
  }
  printf(" The number of unique timesteps/substeps in the data: %d\n",timestep_size);

  // The num_timesteps array contains a list of the unique timesteps found in the data
  // Now sort the num_timesteps array
  double tmp;
  for (i=0;i<timestep_size;i++) {
    for (j=0;j<timestep_size;j++) {
      if (num_timesteps[j] > num_timesteps[i] ) {
        tmp = num_timesteps[i];
        num_timesteps[i] = num_timesteps[j];
        num_timesteps[j] = tmp;
      }
    }
  }


  int ***data;
  data = (int ***) malloc(sizeof(int **)*maxlevel);
  for (i=0;i<maxlevel;i++) {
    data[i] = (int **) malloc(sizeof(int *)*timestep_size);
    for (j=0;j<timestep_size;j++) {
      data[i][j] = (int *) malloc(sizeof(int)*size);
    }
  }

  // bin by level
  int **levels;
  levels = (int **) malloc(sizeof(int *)*maxlevel);
  for (i=0;i<maxlevel;i++) {
    levels[i] = (int *) malloc(sizeof(int)*size);
  }

  int *level_size;
  level_size = (int *) malloc(sizeof(int *)*maxlevel);
  for (i=0;i<maxlevel;i++) {
    level_size[i] = 0;
  }

  for (j=0;j<maxlevel;j++) {
    for (i=0;i<size;i++) {
      if ( level[i] == j ) {
        levels[j][level_size[j]] = i;
        ++level_size[j];
      }
    }
  }

  int **datasize;
  datasize = (int **) malloc(sizeof(int *)*maxlevel);
  for (i=0;i<maxlevel;i++) {
    datasize[i] = (int *) malloc(sizeof(int)*timestep_size);
  }

  for (i=0;i<maxlevel;i++) {
    for (j=0;j<timestep_size;j++) {
      datasize[i][j] = 0;
    }
  }

  // Now bin the data according to timestep
  for (j=0;j<maxlevel;j++) {
    for (i=0;i<level_size[j];i++) {
      for (k=0;k<timestep_size;k++) {
        if ( floatcmp(num_timesteps[k],timesteps[levels[j][i]]) == 1 ) {
          data[j][k][datasize[j][k]] = levels[j][i];
          ++datasize[j][k];
        }
      }
    }
  }

  // Quick test
  //printf(" Num of points %d timestep %g timestep2 %g\n",datasize[2][3],num_timesteps[3],timesteps[data[2][3][0]]);

  // Now we sort the coordinate position for each timestep and level
  double *x_tmp, *x_nod, *data_nod;
  int ii,jj;
  int tmp_size;
  x_tmp = (double *) malloc(sizeof(double)*size);
  x_nod = (double *) malloc(sizeof(double)*size);
  data_nod = (double *) malloc(sizeof(double)*size);
  for (j=0;j<maxlevel;j++) {
    for (k=0;k<timestep_size;k++) {
      for (i=0;i<datasize[j][k];i++) {
        x_tmp[i] = x[ data[j][k][i] ];
      }

      // Sort the temporary array
      for (ii=0;ii<datasize[j][k];ii++) {
        for (jj=0;jj<datasize[j][k];jj++) {
          if (x_tmp[jj] > x_tmp[ii] ) {
            tmp = x_tmp[ii];
            x_tmp[ii] = x_tmp[jj];
            x_tmp[jj] = tmp;
          }
        }
      }

      tmp_size = 0;
      if ( datasize[j][k] > 0 ) {
        x_nod[0] = x_tmp[0];
        tmp_size++;
      }
      // Remove duplicates
      for (i=1;i<datasize[j][k];i++) {
        if ( floatcmp(x_tmp[i-1],x_tmp[i]) ) {
          // Duplicate
          // don't record
        } else {
          x_nod[tmp_size] = x_tmp[i];
          tmp_size++;
        }  
      }

      // put the values in an array
      for (i=0;i<datasize[j][k];i++) {
        for (ii=0;ii<tmp_size;ii++) {
          if ( floatcmp( x[ data[j][k][i] ], x_nod[ii]) ) {
            data_nod[ii] = value[ data[j][k][i] ];
          }
        }
      }

      shape[0] = tmp_size;
      gft_out_full(basename,num_timesteps[k],shape,cnames,1,x_nod,data_nod);

    }
  }

  return 0;
}
