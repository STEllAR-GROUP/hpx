// 27 Sep 2010
// Matt Anderson
// FMR nlsm code for strong scaling comparison with HPX

#include <iostream>
#include <vector>
#include <math.h>
#include <sdf.h>
#include <mpi.h>
using namespace std;

int initial_data(int offset,
                 int numprocs,
                 int myid,
                 std::vector<double> &r,
                 std::vector<double> &chi,
                 std::vector<double> &Phi,
                 std::vector<double> &Pi,
                 std::vector<double> &energy,int,double,int);

int calc_rhs(int numprocs,
             int myid,
             std::vector<double> &r,
             std::vector<double> &chi,
             std::vector<double> &Phi,
             std::vector<double> &Pi,
             std::vector<double> &rhs_chi,
             std::vector<double> &rhs_Phi,
             std::vector<double> &rhs_Pi,int PP,double eps); 

int communicate(int numprocs,
                int myid,
             std::vector<double> &r,
             std::vector<double> &field);

int finer_mesh(int numprocs,
               int myid,
               int timestep,
               double time,
               int gw,
               std::vector< double > *r,
               std::vector< double > *chi,
               std::vector< double > *Phi,
               std::vector< double > *Pi,
               std::vector< double > *chi_np1,
               std::vector< double > *Phi_np1,
               std::vector< double > *Pi_np1,
               std::vector< double > *rhs_chi,
               std::vector< double > *rhs_Phi,
               std::vector< double > *rhs_Pi,
               std::vector< int > *fine_id,
               std::vector< int > *coarse_id,
               std::vector< int > *restrict_coarse,
               std::vector< int > *restrict_fine,
               int *nx,double *dt,
               int *lower, int *upper,
               int *global_nx,
               int gz_offset,
               int PP, double eps, int level);

const int nlevels = 4;

int main(int argc,char* argv[]) {

  int myid,numprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); 

  std::vector< double > r[nlevels];
  std::vector< double > chi[nlevels],chi_np1[nlevels],rhs_chi[nlevels];
  std::vector< double > Phi[nlevels],Phi_np1[nlevels],rhs_Phi[nlevels];
  std::vector< double > Pi[nlevels],Pi_np1[nlevels],rhs_Pi[nlevels];
  std::vector< double > energy[nlevels];

  char cnames[80];
  int i,j,k,count;
  int gw = 10;

  int global_nx[nlevels];
  int nx[nlevels];

  global_nx[0] = 500;
  for (i=1;i<nlevels;i++) {
    global_nx[i] = 141;
  }

  if ( global_nx[0]%numprocs != 0 ) {
    std::cerr << " Problem:  " << global_nx[0] << " not divisible by " << numprocs << std::endl;
    MPI_Finalize();
    return 0;
  }

  nx[0] = global_nx[0]/numprocs;
  for (i=1;i<nlevels;i++) {
    nx[i] = (global_nx[i]-1)/numprocs;
  }

  double t1,t2;
  t1 = MPI_Wtime();

  // CODiss 
  double eps = 0.3;
  int nt0 = 1600;
  double time = 0.0;
  int shape[nlevels][3];
  int PP = 7;

  double dx[nlevels];
  double dt[nlevels];
  dx[0] = 15.0/(global_nx[0]-1);
  dt[0] = 0.15*dx[0];
  for (i=1;i<nlevels;i++) {
    dx[i] = 0.5*dx[i-1];
    dt[i] = 0.5*dt[i-1];
  }

  initial_data(nx[0]*myid,numprocs,myid,r[0],chi[0],Phi[0],Pi[0],energy[0],nx[0],dx[0],PP);

  for (i=1;i<nlevels;i++) {
    if ( numprocs == 1 ) {
      initial_data(nx[i]*myid,numprocs,myid,r[i],chi[i],Phi[i],Pi[i],energy[i],global_nx[i],dx[i],PP);
    } else if ( myid == numprocs-1 ) {
      initial_data(nx[i]*myid,numprocs,myid,r[i],chi[i],Phi[i],Pi[i],energy[i],nx[i]+1,dx[i],PP);
    } else {
      initial_data(nx[i]*myid,numprocs,myid,r[i],chi[i],Phi[i],Pi[i],energy[i],nx[i],dx[i],PP);
    }
  }

  for (i=0;i<nlevels;i++) {
    chi_np1[i].resize(r[i].size());
    Phi_np1[i].resize(r[i].size());
    Pi_np1[i].resize(r[i].size());

    rhs_chi[i].resize(r[i].size());
    rhs_Phi[i].resize(r[i].size());
    rhs_Pi[i].resize(r[i].size());
    shape[i][0]  = r[i].size();
  }

  char filename[80];
  char basename[80];
  sprintf(cnames,"r");

  // account for ghostzones
  int gz_offset = 0;
  if ( myid != 0 ) gz_offset = 3;

  // figure out ghostwidth communication
  std::vector<int> coarse_id[nlevels], fine_id[nlevels];
  for (i=1;i<nlevels;i++) {
    coarse_id[i].resize(gw/2);
    fine_id[i].resize(gw/2);
    for (j=0;j<gw/2;j++) {
      // initialize
      coarse_id[i][j] = -1;
      fine_id[i][j] = -1;

      int m,mm;
      m  = global_nx[i]-2*j;
      mm = (m-1)/2;
  
      for (k=0;k<numprocs;k++) {
        
        // figure out which processor has the 'mm' index
        if ( mm >= k*nx[i-1] && ( (mm < (k+1)*nx[i-1] && k < numprocs-1 ) || k == numprocs-1) ) {
          coarse_id[i][j] = k;
          //std::cout << " processor " << myid << " has " << mm << std::endl;
        }

        // figure out which processor has the 'm' index
        if ( m-1 >= k*nx[i] && ( ( m-1 < (k+1)*nx[i] && k < numprocs-1 ) ||  k == numprocs-1 ) ) {
          //std::cout << " processor " << myid << " has level 1 index " << m-1 << std::endl;
          fine_id[i][j] = k;
        }
      }
      //std::cout << " myid " << myid << " coarse id " << coarse_id[i][j] << " fine id " << fine_id[i][j] << " j " << j << " m " << m << " mm " << mm << " level " << i << std::endl;
    }
  }

  // figure out injection communication
  // injecting -- overwrite coarse mesh points with finer mesh result

  std::vector<int> restrict_coarse[nlevels], restrict_fine[nlevels];
  for (i=1;i<nlevels;i++) {
    count = 0;

    restrict_coarse[i].resize(global_nx[i]-10);
    restrict_fine[i].resize(global_nx[i]-10);

    // initialize
    for (j=0;j<restrict_coarse[i].size();j++) {
      restrict_coarse[i][j] = -1;
      restrict_fine[i][j] = -1;
    }

    for (j=0;j<global_nx[i]-10;j = j+2) {
      for (k=0;k<numprocs;k++) {
        //figure out which processor j is on
        if ( j >= k*nx[i] && ( ( j < (k+1)*nx[i] && k < numprocs-1 ) ||  k == numprocs-1 ) ) {
        //  std::cout << " index " << j << " is on " << k << std::endl;
          restrict_fine[i][j] = k;
        }
        //figure out which processor count is on
        if ( count >= k*nx[i-1] && ( ( count < (k+1)*nx[i-1] && k < numprocs-1 ) || k == numprocs-1 ) ) {
        //  std::cout << " count " << count << " is on " << k << std::endl;
          restrict_coarse[i][count] = k;
        }
      }
  
      count++;
    } 
  }
  // finished with the injection communication setup

  // update rhs over non-ghostzone sites
  int lower[nlevels],upper[nlevels];
  for (i=0;i<nlevels;i++) {
    if (myid == 0 && numprocs == 1) {
      lower[i] = 1;
      upper[i] = r[i].size();
    } else if (myid == 0) {
      lower[i] = 1;
      upper[i] = r[i].size()-3;
    } else if ( myid == numprocs-1 ) {
      lower[i] = 3;
      upper[i] = r[i].size();
    } else {
      lower[i] = 3;
      upper[i] = r[i].size()-3;
    }
  }

  for (i=0;i<nt0;i++) {

     // Coarse mesh evolution {{{
     // ------------------------------- iter 1
     calc_rhs(numprocs,myid,r[0],chi[0],Phi[0],Pi[0],rhs_chi[0],rhs_Phi[0],rhs_Pi[0],PP,eps);

     for (j=lower[0];j<upper[0];j++) {
       chi_np1[0][j] = chi[0][j] + rhs_chi[0][j]*dt[0]; 
       Phi_np1[0][j] = Phi[0][j] + rhs_Phi[0][j]*dt[0]; 
       Pi_np1[0][j] =  Pi[0][j] + rhs_Pi[0][j]*dt[0]; 

     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0][0] = 4./3*chi_np1[0][1] -1./3*chi_np1[0][2];
       Pi_np1[0][0]  = 4./3*Pi_np1[0][1]  -1./3*Pi_np1[0][2];
       Phi_np1[0][1] = 0.5*Phi_np1[0][2];
     }

     //---------------------------------- iter 2
     communicate(numprocs,myid,r[0],chi_np1[0]);
     communicate(numprocs,myid,r[0],Phi_np1[0]);
     communicate(numprocs,myid,r[0],Pi_np1[0]);
     calc_rhs(numprocs,myid,r[0],chi_np1[0],Phi_np1[0],Pi_np1[0],rhs_chi[0],rhs_Phi[0],rhs_Pi[0],PP,eps);

     for (j=lower[0];j<upper[0];j++) {
       chi_np1[0][j] = 0.75*chi[0][j] + 0.25*chi_np1[0][j] + 0.25*rhs_chi[0][j]*dt[0]; 
       Phi_np1[0][j] = 0.75*Phi[0][j] + 0.25*Phi_np1[0][j] + 0.25*rhs_Phi[0][j]*dt[0]; 
       Pi_np1[0][j]  = 0.75*Pi[0][j]  + 0.25*Pi_np1[0][j]  + 0.25*rhs_Pi[0][j]*dt[0]; 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0][0] = 4./3*chi_np1[0][1] -1./3*chi_np1[0][2];
       Pi_np1[0][0]  = 4./3*Pi_np1[0][1]  -1./3*Pi_np1[0][2];
       Phi_np1[0][1] = 0.5*Phi_np1[0][2];
     }

     //---------------------------------- iter 3
     communicate(numprocs,myid,r[0],chi_np1[0]);
     communicate(numprocs,myid,r[0],Phi_np1[0]);
     communicate(numprocs,myid,r[0],Pi_np1[0]);
     calc_rhs(numprocs,myid,r[0],chi_np1[0],Phi_np1[0],Pi_np1[0],rhs_chi[0],rhs_Phi[0],rhs_Pi[0],PP,eps);

     for (j=lower[0];j<upper[0];j++) {
       chi_np1[0][j] = 1./3*chi[0][j] + 2./3*(chi_np1[0][j] + rhs_chi[0][j]*dt[0]); 
       Phi_np1[0][j] = 1./3*Phi[0][j] + 2./3*(Phi_np1[0][j] + rhs_Phi[0][j]*dt[0]); 
       Pi_np1[0][j]  = 1./3*Pi[0][j]  + 2./3*(Pi_np1[0][j]  + rhs_Pi[0][j]*dt[0]); 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0][0] = 4./3*chi_np1[0][1] -1./3*chi_np1[0][2];
       Pi_np1[0][0]  = 4./3*Pi_np1[0][1]  -1./3*Pi_np1[0][2];
       Phi_np1[0][1] = 0.5*Phi_np1[0][2];
     }

     // }}}

    finer_mesh(numprocs,myid,i,i*dt[0],gw,r,chi,Phi,Pi,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,
               fine_id,coarse_id,restrict_coarse,restrict_fine,nx,dt,lower,upper,global_nx,gz_offset,PP,eps,1);

    chi[0].swap(chi_np1[0]);
    Phi[0].swap(Phi_np1[0]);
    Pi[0].swap(Pi_np1[0]);

    communicate(numprocs,myid,r[0],chi[0]);
    communicate(numprocs,myid,r[0],Phi[0]);
    communicate(numprocs,myid,r[0],Pi[0]);

    sprintf(basename,"chi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt[0]*(i+1),shape[0],cnames,1,&*r[0].begin(),&*chi[0].begin());
    sprintf(basename,"Phi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt[0]*(i+1),shape[0],cnames,1,&*r[0].begin(),&*Phi[0].begin());
    sprintf(basename,"Pi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt[0]*(i+1),shape[0],cnames,1,&*r[0].begin(),&*Pi[0].begin());
  }
  t2 = MPI_Wtime();
  if ( myid == 0 ) {
    std::cout << " Elapsed time: " << t2-t1 << std::endl;
  }

  MPI_Finalize();
  return 0;
}

int finer_mesh(int numprocs,
               int myid,
               int timestep,
               double time,
               int gw,
               std::vector< double > *r,
               std::vector< double > *chi,
               std::vector< double > *Phi,
               std::vector< double > *Pi,
               std::vector< double > *chi_np1,
               std::vector< double > *Phi_np1,
               std::vector< double > *Pi_np1,
               std::vector< double > *rhs_chi,
               std::vector< double > *rhs_Phi,
               std::vector< double > *rhs_Pi,
               std::vector< int > *fine_id,
               std::vector< int > *coarse_id,
               std::vector< int > *restrict_coarse,
               std::vector< int > *restrict_fine,
               int *nx, double *dt,
               int *lower, int *upper,
               int *global_nx,
               int gz_offset,
               int PP, double eps, int level)
{

    // local variables
    int j,k;
    char filename[80];
    char basename[80];
    char cnames[80];
    sprintf(cnames,"r");
    // MPI auxiliary variables
    double buffer[6],rbuffer[6];
    int tag = 98;
    int tag2 = 97;
    MPI_Status status;
    MPI_Request request;
    int shape[3];
    shape[0] = r[level].size();

    // fill ghostzones {{{
    if ( timestep != 0 && 1==1) {
      // fill the last gw points in the finer mesh using the coarse mesh 
      for (j=0;j<gw/2;j++) {
        int m,mm;
        m  = global_nx[level]-2*j;
        mm = (m-1)/2;

        if ( fine_id[level][j] == -1 || coarse_id[level][j] == -1 ) {
          std::cout << " PROBLEM " << m << " " << mm << std::endl;
          exit(0);
        }

        if ( coarse_id[level][j] == fine_id[level][j] && myid == fine_id[level][j] ) {
          // no communication necessary
          chi[level][m-1-myid*nx[level]+gz_offset ] = chi[level-1][mm-myid*nx[level-1]+gz_offset];
          Phi[level][m-1-myid*nx[level]+gz_offset ] = Phi[level-1][mm-myid*nx[level-1]+gz_offset];
          Pi[level][m-1-myid*nx[level]+gz_offset ] = Pi[level-1][mm-myid*nx[level-1]+gz_offset];
          chi[level][m-myid*nx[level]+gz_offset] = 0.5*chi[level-1][mm-myid*nx[level-1]+gz_offset] 
                                                 + 0.5*chi[level-1][mm+1-myid*nx[level-1]+gz_offset];
          Phi[level][m-myid*nx[level]+gz_offset] = 0.5*Phi[level-1][mm-myid*nx[level-1]+gz_offset] 
                                                 + 0.5*Phi[level-1][mm+1-myid*nx[level-1]+gz_offset];
          Pi[level][m-myid*nx[level]+gz_offset] = 0.5*Pi[level-1][mm-myid*nx[level-1]+gz_offset] 
                                                + 0.5*Pi[level-1][mm+1-myid*nx[level-1]+gz_offset];
        } else {
          if ( myid == coarse_id[level][j] ) {
            // send info to fine_id
            buffer[0] = chi[level-1][mm-myid*nx[level-1]+gz_offset];
            buffer[1] = Phi[level-1][mm-myid*nx[level-1]+gz_offset];
            buffer[2] = Pi[level-1][mm-myid*nx[level-1]+gz_offset];
            buffer[3] = chi[level-1][mm+1-myid*nx[level-1]+gz_offset];
            buffer[4] = Phi[level-1][mm+1-myid*nx[level-1]+gz_offset];
            buffer[5] = Pi[level-1][mm+1-myid*nx[level-1]+gz_offset];
            MPI_Send(buffer,6,MPI_DOUBLE,fine_id[level][j],tag,MPI_COMM_WORLD);
          } else if ( myid == fine_id[level][j] ) {
            // receive info from coarse_id
            MPI_Recv(rbuffer,6,MPI_DOUBLE,coarse_id[level][j],tag,MPI_COMM_WORLD,&status);
            chi[level][m-1-myid*nx[level]+gz_offset] = rbuffer[0];
            Phi[level][m-1-myid*nx[level]+gz_offset] = rbuffer[1];
            Pi[level][m-1-myid*nx[level]+gz_offset] = rbuffer[2];
            chi[level][m-myid*nx[level]+gz_offset] = 0.5*rbuffer[0]+0.5*rbuffer[3];
            Phi[level][m-myid*nx[level]+gz_offset] = 0.5*rbuffer[1]+0.5*rbuffer[4];
            Pi[level][m-myid*nx[level]+gz_offset] = 0.5*rbuffer[2]+0.5*rbuffer[5];
          }
        }

      }
    }
    // }}}
    // take two steps of the finer mesh
    for (k=0;k<2;k++) {
     // Finer mesh evolution {{{
     // ------------------------------- iter 1
     // communicate because of filling ghostzones
     communicate(numprocs,myid,r[level],chi[level]);
     communicate(numprocs,myid,r[level],Phi[level]);
     communicate(numprocs,myid,r[level],Pi[level]);
     calc_rhs(numprocs,myid,r[level],chi[level],Phi[level],Pi[level],rhs_chi[level],rhs_Phi[level],rhs_Pi[level],PP,eps);

     for (j=lower[level];j<upper[level];j++) {
       chi_np1[level][j] = chi[level][j] + rhs_chi[level][j]*dt[level]; 
       Phi_np1[level][j] = Phi[level][j] + rhs_Phi[level][j]*dt[level]; 
       Pi_np1[level][j] =  Pi[level][j] + rhs_Pi[level][j]*dt[level]; 

     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[level][0] = 4./3*chi_np1[level][1] -1./3*chi_np1[level][2];
       Pi_np1[level][0]  = 4./3*Pi_np1[level][1]  -1./3*Pi_np1[level][2];
       Phi_np1[level][1] = 0.5*Phi_np1[level][2];
     }

     //---------------------------------- iter 2
     communicate(numprocs,myid,r[level],chi_np1[level]);
     communicate(numprocs,myid,r[level],Phi_np1[level]);
     communicate(numprocs,myid,r[level],Pi_np1[level]);
     calc_rhs(numprocs,myid,r[level],chi_np1[level],Phi_np1[level],Pi_np1[level],rhs_chi[level],rhs_Phi[level],rhs_Pi[level],PP,eps);

     for (j=lower[level];j<upper[level];j++) {
       chi_np1[level][j] = 0.75*chi[level][j] + 0.25*chi_np1[level][j] + 0.25*rhs_chi[level][j]*dt[level]; 
       Phi_np1[level][j] = 0.75*Phi[level][j] + 0.25*Phi_np1[level][j] + 0.25*rhs_Phi[level][j]*dt[level]; 
       Pi_np1[level][j]  = 0.75*Pi[level][j]  + 0.25*Pi_np1[level][j]  + 0.25*rhs_Pi[level][j]*dt[level]; 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[level][0] = 4./3*chi_np1[level][1] -1./3*chi_np1[level][2];
       Pi_np1[level][0]  = 4./3*Pi_np1[level][1]  -1./3*Pi_np1[level][2];
       Phi_np1[level][1] = 0.5*Phi_np1[level][2];
     }

     //---------------------------------- iter 3
     communicate(numprocs,myid,r[level],chi_np1[level]);
     communicate(numprocs,myid,r[level],Phi_np1[level]);
     communicate(numprocs,myid,r[level],Pi_np1[level]);
     calc_rhs(numprocs,myid,r[level],chi_np1[level],Phi_np1[level],Pi_np1[level],rhs_chi[level],rhs_Phi[level],rhs_Pi[level],PP,eps);

     for (j=lower[level];j<upper[level];j++) {
       chi_np1[level][j] = 1./3*chi[level][j] + 2./3*(chi_np1[level][j] + rhs_chi[level][j]*dt[level]); 
       Phi_np1[level][j] = 1./3*Phi[level][j] + 2./3*(Phi_np1[level][j] + rhs_Phi[level][j]*dt[level]); 
       Pi_np1[level][j]  = 1./3*Pi[level][j]  + 2./3*(Pi_np1[level][j]  + rhs_Pi[level][j]*dt[level]); 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[level][0] = 4./3*chi_np1[level][1] -1./3*chi_np1[level][2];
       Pi_np1[level][0]  = 4./3*Pi_np1[level][1]  -1./3*Pi_np1[level][2];
       Phi_np1[level][1] = 0.5*Phi_np1[level][2];
     }

     // }}}

      if ( level < nlevels-1 ) {
        finer_mesh(numprocs,myid,timestep+k,time + k*dt[level],gw,r,chi,Phi,Pi,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,
                 fine_id,coarse_id,restrict_coarse,restrict_fine,nx,dt,lower,upper,global_nx,gz_offset,PP,eps,level+1);
      }

      chi[level].swap(chi_np1[level]);
      Phi[level].swap(Phi_np1[level]);
      Pi[level].swap(Pi_np1[level]);

      communicate(numprocs,myid,r[level],chi[level]);
      communicate(numprocs,myid,r[level],Phi[level]);
      communicate(numprocs,myid,r[level],Pi[level]);

      sprintf(basename,"chi%d",level);
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,time+(k+1)*dt[level],shape,cnames,1,&*r[level].begin(),&*chi[level].begin());
      sprintf(basename,"Phi%d",level);
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,time+(k+1)*dt[level],shape,cnames,1,&*r[level].begin(),&*Phi[level].begin());
      sprintf(basename,"Pi%d",level);
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,time+(k+1)*dt[level],shape,cnames,1,&*r[level].begin(),&*Pi[level].begin());
    }

    // injecting -- overwrite coarse mesh points with finer mesh result {{{
    int count = 0;
    for (j=0;j<global_nx[level]-10;j = j+2) {
      if ( restrict_coarse[level][count] == -1 || restrict_fine[level][j] == -1 ) {
        std::cerr << " Problem " << std::endl;
        exit(0);
      }

      if ( restrict_coarse[level][count] == restrict_fine[level][j] && myid == restrict_fine[level][j] ) {
        // no communication needed for restriction
         chi_np1[level-1][count-myid*nx[level-1]+gz_offset] = chi[level][j-myid*nx[level]+gz_offset];
         Phi_np1[level-1][count-myid*nx[level-1]+gz_offset] = Phi[level][j-myid*nx[level]+gz_offset];
         Pi_np1[level-1][count-myid*nx[level-1]+gz_offset] = Pi[level][j-myid*nx[level]+gz_offset];
      } else {
        if ( myid == restrict_coarse[level][count] ) {
           MPI_Recv(rbuffer,3,MPI_DOUBLE,restrict_fine[level][j],tag2,MPI_COMM_WORLD,&status);
           chi_np1[level-1][count-myid*nx[level-1]+gz_offset] = rbuffer[0];
           Phi_np1[level-1][count-myid*nx[level-1]+gz_offset] = rbuffer[1];
           Pi_np1[level-1][count-myid*nx[level-1]+gz_offset] = rbuffer[2];
        } else if ( myid == restrict_fine[level][j] ) {
           buffer[0] = chi[level][j-myid*nx[level]+gz_offset];
           buffer[1] = Phi[level][j-myid*nx[level]+gz_offset];
           buffer[2] = Pi[level][j-myid*nx[level]+gz_offset];
           MPI_Send(buffer,3,MPI_DOUBLE,restrict_coarse[level][count],tag2,MPI_COMM_WORLD);
        }
      }
      count++;
    } 
    // }}}

    return 0;
}

//
//
//
//
//
int initial_data(int offset,
                 int numprocs,
                 int myid,
                 std::vector<double> &r,
                 std::vector<double> &chi,
                 std::vector<double> &Phi,
                 std::vector<double> &Pi,
                 std::vector<double> &energy,int nx0,double dx,int PP) {


  int i;

  // add ghostzones
  if ( myid != 0 ) {
    r.push_back((offset-3)*dx);
    r.push_back((offset-2)*dx);
    r.push_back((offset-1)*dx);
  }

  for (i=0;i<nx0;i++) {
    r.push_back((offset+i)*dx);
  }

  // add ghostzones
  if ( myid != (numprocs-1) ) {
    r.push_back((offset+nx0-1+1)*dx);
    r.push_back((offset+nx0-1+2)*dx);
    r.push_back((offset+nx0-1+3)*dx);
  }

  double amp = 0.01;
  double delta = 1.0;
  double R0 = 8.0;

  for (i=0;i<r.size();i++) {
    chi.push_back(amp*exp(-(r[i]-R0)*(r[i]-R0)/(delta*delta)));  
    Phi.push_back(amp*exp(-(r[i]-R0)*(r[i]-R0)/(delta*delta)) * ( -2.*(r[i]-R0)/(delta*delta)  )   );
    Pi.push_back(0.0);
    energy.push_back( 0.5*r[i]*r[i]*(Pi[i]*Pi[i] + Phi[i]*Phi[i])-r[i]*r[i]*pow(chi[i],PP+1)/(PP+1) );
  }

  return 0;
}

int calc_rhs(int numprocs,
             int myid,
             std::vector<double> &r,
             std::vector<double> &chi,
             std::vector<double> &Phi,
             std::vector<double> &Pi,
             std::vector<double> &rhs_chi,
             std::vector<double> &rhs_Phi,
             std::vector<double> &rhs_Pi,int PP,double eps) {

 
  int i;

  double dr = r[1] - r[0]; 

  std::vector<double> diss_chi,diss_Phi,diss_Pi;
  diss_chi.resize(r.size()); diss_Phi.resize(r.size()); diss_Pi.resize(r.size());
  for (i=0;i<r.size();i++) {
    if ( i>= 3 && i < r.size()-3 ) {
      diss_chi[i] = -1./(64.*dr)*(    -chi[i-3] 
                                + 6.*chi[i-2]
                                -15.*chi[i-1]
                                +20.*chi[i] 
                                -15.*chi[i+1]
                                 +6.*chi[i+2]
                                    -chi[i+3] );
      diss_Phi[i] = -1./(64.*dr)*(    -Phi[i-3] 
                                + 6.*Phi[i-2]
                                -15.*Phi[i-1]
                                +20.*Phi[i] 
                                -15.*Phi[i+1]
                                 +6.*Phi[i+2]
                                    -Phi[i+3] );
      diss_Pi[i] = -1./(64.*dr)*(    -Pi[i-3] 
                               + 6.*Pi[i-2]
                               -15.*Pi[i-1]
                               +20.*Pi[i] 
                               -15.*Pi[i+1]
                                +6.*Pi[i+2]
                                   -Pi[i+3] );
    } else {
      diss_chi[i] = 0.0;
      diss_Phi[i] = 0.0;
      diss_Pi[i] = 0.0;
    }
  }

  // TEST busy work to test scaling
  //int maxnum = (int) 1000*r.size();
  //for (int i=0;i<maxnum;i++) {
  //  diss_chi[0] += i*(diss_Phi[0]+diss_chi[0])/maxnum;
  //}

  int lower;
  int upper;

  if ( myid == 0 && numprocs == 1 ) {
    lower = 1;
    upper = r.size()-1;
  } else if (myid == 0) {
    lower = 1;
    upper = r.size()-3;
  } else if ( myid == numprocs-1 ) {
    lower = 3;
    upper = r.size()-1;
  } else {
    lower = 3;
    upper = r.size()-3;
  }

  for (i=lower;i<upper;i++) {
    rhs_chi[i] = Pi[i] + eps*diss_chi[i];
    rhs_Phi[i] = 1./(2.*dr) * ( Pi[i+1] - Pi[i-1] ) + eps*diss_Phi[i];
    rhs_Pi[i]  = 3 * ( r[i+1]*r[i+1]*Phi[i+1] - r[i-1]*r[i-1]*Phi[i-1] )/( pow(r[i+1],3) - pow(r[i-1],3) )
                    + 0.0*pow(chi[i],PP) + eps*diss_Pi[i]; // + 1.e-8*diss_chi[0];
  }

  //std::cout << " TEST right boundary " << r.size()-1 << std::endl;
  if ( myid == numprocs-1 ) {
    i = r.size()-1;
    rhs_chi[i] = Pi[i];
    rhs_Phi[i] = -(3.*Phi[i] - 4.*Phi[i-1] + Phi[i-2])/(2.*dr) - Phi[i]/r[i];
    rhs_Pi[i]  = -Pi[i]/r[i] - (3.*Pi[i] - 4.*Pi[i-1] + Pi[i-2])/(2.*dr);
  }

  return 0;
}

int communicate(int numprocs,
                int myid,
             std::vector<double> &r,
             std::vector<double> &field)
{

  double buffer[3],buffer1[3],buffer2[3],rbuffer[3],rbuffer2[3];
  MPI_Status status;
  MPI_Request request;
  int tag = 99;
  int j;

  if ( numprocs > 1 ) {
    if (myid == 0) {
      for (j=0;j<3;j++) buffer[j] = field[r.size()-6+j];
    } else if (myid == numprocs-1) {
      for (j=0;j<3;j++) buffer[j] = field[j+3];
    } else {
      for (j=0;j<3;j++) buffer1[j] = field[j+3];
      for (j=0;j<3;j++) buffer2[j] = field[r.size()-6+j];
    }    

    MPI_Request request;
    int tag = 99;
    if (myid == 0) {
      MPI_Send(buffer,3,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD);
    } else if ( myid == numprocs-1 ) {
      MPI_Recv(rbuffer,3,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
    } else {
      MPI_Isend(buffer2,3,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD,&request);
      MPI_Recv(rbuffer,3,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
      MPI_Wait(&request,&status);
    }

    // communicate other way now
    if (myid == 0) {
      MPI_Recv(rbuffer,3,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD,&status);
    } else if ( myid == numprocs-1 ) {
      MPI_Send(buffer,3,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD);
    } else {
      MPI_Isend(buffer1,3,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&request);
      MPI_Recv(rbuffer2,3,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD,&status);
      MPI_Wait(&request,&status);
    }

    if (myid == 0) {
      field[r.size()-3] = rbuffer[0];
      field[r.size()-2] = rbuffer[1];
      field[r.size()-1] = rbuffer[2];
    } else if ( myid == numprocs-1 ) {
      field[0] = rbuffer[0];
      field[1] = rbuffer[1];
      field[2] = rbuffer[2];
    } else {
      field[0] = rbuffer[0];
      field[1] = rbuffer[1];
      field[2] = rbuffer[2];

      field[r.size()-3] = rbuffer2[0];
      field[r.size()-2] = rbuffer2[1];
      field[r.size()-1] = rbuffer2[2];
    }

  }

  return 0;
}
