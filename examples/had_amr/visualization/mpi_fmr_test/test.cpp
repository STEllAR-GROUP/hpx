
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

int main(int argc,char* argv[]) {

  int myid,numprocs;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); 

  std::vector< double > r;
  std::vector< double > chi,chi_np1,rhs_chi;
  std::vector< double > Phi,Phi_np1,rhs_Phi;
  std::vector< double > Pi,Pi_np1,rhs_Pi;
  std::vector< double > energy;

  std::vector< double > _r;
  std::vector< double > _chi,_chi_np1,_rhs_chi;
  std::vector< double > _Phi,_Phi_np1,_rhs_Phi;
  std::vector< double > _Pi,_Pi_np1,_rhs_Pi;
  std::vector< double > _energy;
  char cnames[80];

  int i,j,k,count;
  int gw = 10;

  int global_nx0 = 500;

  int global_nx1 = 141;

  if ( global_nx1%2 == 0 ) {
    std::cerr << " Problem:  nx1 must be odd " << std::endl;
    exit(0);
  }

  if ( global_nx0%numprocs != 0 || (global_nx1-1)%numprocs != 0 ) {
    std::cerr << " Problem:  either " << global_nx0 << " not divisible by " << numprocs << std::endl;
    std::cerr << "               or " << global_nx1 << " not divisible by " << numprocs << std::endl;
    MPI_Finalize();
    return 0;
  }

  int nx0 = global_nx0/numprocs;
  int offset = myid*nx0;

  int nx1 = (global_nx1-1)/numprocs;
  int offset1 = myid*nx1;

  double t1,t2;
  t1 = MPI_Wtime();

  // CODiss 
  double eps = 0.3;
  int nt0 = 1600;
  double time = 0.0;
  int shape[3];
  int shape1[3];
  int PP = 7;

  double dx = 15.0/(global_nx0-1);
  double dt = 0.15*dx;
  double dt1 = 0.5*dt;
  double dx1 = 0.5*dx;

  initial_data(offset,numprocs,myid,r,chi,Phi,Pi,energy,nx0,dx,PP);

  if ( numprocs == 1 ) {
    initial_data(offset1,numprocs,myid,_r,_chi,_Phi,_Pi,_energy,global_nx1,dx1,PP);
  } else if ( myid == numprocs-1 ) {
    initial_data(offset1,numprocs,myid,_r,_chi,_Phi,_Pi,_energy,nx1+1,dx1,PP);
  } else {
    initial_data(offset1,numprocs,myid,_r,_chi,_Phi,_Pi,_energy,nx1,dx1,PP);
  }

  chi_np1.resize(r.size());
  Phi_np1.resize(r.size());
  Pi_np1.resize(r.size());

  rhs_chi.resize(r.size());
  rhs_Phi.resize(r.size());
  rhs_Pi.resize(r.size());

  _chi_np1.resize(_r.size());
  _Phi_np1.resize(_r.size());
  _Pi_np1.resize(_r.size());

  _rhs_chi.resize(_r.size());
  _rhs_Phi.resize(_r.size());
  _rhs_Pi.resize(_r.size());

  char filename[80];
  char basename[80];
  shape[0]  = r.size();
  shape1[0]  = _r.size();
  sprintf(cnames,"r");
#if 0
  sprintf(basename,"r");
  sprintf(filename,"%d%s",myid,basename);
  gft_out_full(filename,time,shape,cnames,1,&*r.begin(),&*r.begin());
  sprintf(basename,"chi0");
  sprintf(filename,"%d%s",myid,basename);
  gft_out_full(filename,time,shape,cnames,1,&*r.begin(),&*chi.begin());
  sprintf(basename,"Phi0");
  sprintf(filename,"%d%s",myid,basename);
  gft_out_full(filename,time,shape,cnames,1,&*r.begin(),&*Phi.begin());
  sprintf(basename,"Pi0");
  sprintf(filename,"%d%s",myid,basename);
  gft_out_full(filename,time,shape,cnames,1,&*r.begin(),&*Pi.begin());
#endif

  // account for ghostzones
  int gz_offset = 0;
  if ( myid != 0 ) gz_offset = 3;

  // figure out ghostwidth communication
  std::vector<int> coarse_id, fine_id;
  coarse_id.resize(gw/2);
  fine_id.resize(gw/2);
  for (j=0;j<gw/2;j++) {
    // initialize
    coarse_id[j] = -1;
    fine_id[j] = -1;

    int m,mm;
    m  = global_nx1-2*j;
    mm = (m-1)/2;

    for (k=0;k<numprocs;k++) {
        
      // figure out which processor has the 'mm' index
      if ( mm > k*nx0 && mm < (k+1)*nx0 ) {
        coarse_id[j] = k;
        //std::cout << " processor " << myid << " has " << mm << std::endl;
      }

      // figure out which processor has the 'm' index
      if ( m-1 > k*nx1 && ( ( m-1 < (k+1)*nx1 && k < numprocs-1 ) ||  k == numprocs-1 ) ) {
        //std::cout << " processor " << myid << " has level 1 index " << m-1 << std::endl;
        fine_id[j] = k;
      }
    }
    //std::cout << " myid " << myid << " coarse id " << coarse_id[j] << " fine id " << fine_id[j] << " j " << j << " m " << m << std::endl;
  }

  // figure out injection communication
  // injecting -- overwrite coarse mesh points with finer mesh result
  count = 0;

  std::vector<int> restrict_coarse, restrict_fine;
  restrict_coarse.resize(global_nx1-10);
  restrict_fine.resize(global_nx1-10);

  // initialize
  for (j=0;j<restrict_coarse.size();j++) {
    restrict_coarse[j] = -1;
    restrict_fine[j] = -1;
  }

  for (j=0;j<global_nx1-10;j = j+2) {
    for (k=0;k<numprocs;k++) {
      //figure out which processor j is on
      if ( j >= k*nx1 && ( ( j < (k+1)*nx1 && k < numprocs-1 ) ||  k == numprocs-1 ) ) {
      //  std::cout << " index " << j << " is on " << k << std::endl;
        restrict_fine[j] = k;
      }
      //figure out which processor count is on
      if ( count >= k*nx0 && count < (k+1)*nx0 ) {
      //  std::cout << " count " << count << " is on " << k << std::endl;
        restrict_coarse[count] = k;
      }
    }
    //if ( myid == 0 ) std::cout << " count " << count << " restrict_coarse " << restrict_coarse[count] << " j " << j <<  " restrict_fine " << restrict_fine[j] << std::endl;
  
    count++;
  } 
  // finished with the injection communication setup


  // update rhs over non-ghostzone sites
  int lower,upper;
  int _lower,_upper;
  if (myid == 0 && numprocs == 1) {
    lower = 1;
    upper = r.size();
    _lower = 1;
    _upper = _r.size();
  } else if (myid == 0) {
    lower = 1;
    upper = r.size()-3;
    _lower = 1;
    _upper = _r.size()-3;
  } else if ( myid == numprocs-1 ) {
    lower = 3;
    upper = r.size();
    _lower = 3;
    _upper = _r.size();
  } else {
    lower = 3;
    upper = r.size()-3;
    _lower = 3;
    _upper = _r.size()-3;
  }

  // MPI auxiliary variables
  double buffer[6],rbuffer[6];
  int tag = 98;
  int tag2 = 97;
  MPI_Status status;
  MPI_Request request;

  for (i=0;i<nt0;i++) {
   // double tA = MPI_Wtime();
     // Coarse mesh evolution {{{
     // ------------------------------- iter 1
     calc_rhs(numprocs,myid,r,chi,Phi,Pi,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     for (j=lower;j<upper;j++) {
       chi_np1[j] = chi[j] + rhs_chi[j]*dt; 
       Phi_np1[j] = Phi[j] + rhs_Phi[j]*dt; 
       Pi_np1[j] =  Pi[j] + rhs_Pi[j]*dt; 

     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
       Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
       Phi_np1[1] = 0.5*Phi_np1[2];
     }

     //---------------------------------- iter 2
     communicate(numprocs,myid,r,chi_np1);
     communicate(numprocs,myid,r,Phi_np1);
     communicate(numprocs,myid,r,Pi_np1);
     calc_rhs(numprocs,myid,r,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     for (j=lower;j<upper;j++) {
       chi_np1[j] = 0.75*chi[j] + 0.25*chi_np1[j] + 0.25*rhs_chi[j]*dt; 
       Phi_np1[j] = 0.75*Phi[j] + 0.25*Phi_np1[j] + 0.25*rhs_Phi[j]*dt; 
       Pi_np1[j]  = 0.75*Pi[j]  + 0.25*Pi_np1[j]  + 0.25*rhs_Pi[j]*dt; 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
       Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
       Phi_np1[1] = 0.5*Phi_np1[2];
     }

     //---------------------------------- iter 3
     communicate(numprocs,myid,r,chi_np1);
     communicate(numprocs,myid,r,Phi_np1);
     communicate(numprocs,myid,r,Pi_np1);
     calc_rhs(numprocs,myid,r,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     for (j=lower;j<upper;j++) {
       chi_np1[j] = 1./3*chi[j] + 2./3*(chi_np1[j] + rhs_chi[j]*dt); 
       Phi_np1[j] = 1./3*Phi[j] + 2./3*(Phi_np1[j] + rhs_Phi[j]*dt); 
       Pi_np1[j]  = 1./3*Pi[j]  + 2./3*(Pi_np1[j]  + rhs_Pi[j]*dt); 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
       Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
       Phi_np1[1] = 0.5*Phi_np1[2];
     }

     // }}}
   // double tB = MPI_Wtime();
   // std::cout << " Elapsed time: " << tB-tA << std::endl;
//#if 0
    // fill ghostzones {{{
    if ( i != 0 && 1==1) {
      // fill the last gw points in the finer mesh using the coarse mesh 
      for (j=0;j<gw/2;j++) {
        int m,mm;
        m  = global_nx1-2*j;
        mm = (m-1)/2;

        if ( fine_id[j] == -1 || coarse_id[j] == -1 ) {
          std::cout << " PROBLEM " << std::endl;
          exit(0);
        }

        if ( coarse_id[j] == fine_id[j] && myid == fine_id[j] ) {
          // no communication necessary
          _chi[m-1-myid*nx1+gz_offset ] = chi[mm-myid*nx0+gz_offset];
          _Phi[m-1-myid*nx1+gz_offset ] = Phi[mm-myid*nx0+gz_offset];
          _Pi[m-1-myid*nx1+gz_offset ] = Pi[mm-myid*nx0+gz_offset];
          _chi[m-myid*nx1+gz_offset] = 0.5*chi[mm-myid*nx0+gz_offset] + 0.5*chi[mm+1-myid*nx0+gz_offset];
          _Phi[m-myid*nx1+gz_offset] = 0.5*Phi[mm-myid*nx0+gz_offset] + 0.5*Phi[mm+1-myid*nx0+gz_offset];
          _Pi[m-myid*nx1+gz_offset] = 0.5*Pi[mm-myid*nx0+gz_offset] + 0.5*Pi[mm+1-myid*nx0+gz_offset];
        } else {
          if ( myid == coarse_id[j] ) {
            // send info to fine_id
            buffer[0] = chi[mm-myid*nx0+gz_offset];
            buffer[1] = Phi[mm-myid*nx0+gz_offset];
            buffer[2] = Pi[mm-myid*nx0+gz_offset];
            buffer[3] = chi[mm+1-myid*nx0+gz_offset];
            buffer[4] = Phi[mm+1-myid*nx0+gz_offset];
            buffer[5] = Pi[mm+1-myid*nx0+gz_offset];
            MPI_Send(buffer,6,MPI_DOUBLE,fine_id[j],tag,MPI_COMM_WORLD);
          } else if ( myid == fine_id[j] ) {
            // receive info from coarse_id
            MPI_Recv(rbuffer,6,MPI_DOUBLE,coarse_id[j],tag,MPI_COMM_WORLD,&status);
            _chi[m-1-myid*nx1+gz_offset] = rbuffer[0];
            _Phi[m-1-myid*nx1+gz_offset] = rbuffer[1];
            _Pi[m-1-myid*nx1+gz_offset] = rbuffer[2];
            _chi[m-myid*nx1+gz_offset] = 0.5*rbuffer[0]+0.5*rbuffer[3];
            _Phi[m-myid*nx1+gz_offset] = 0.5*rbuffer[1]+0.5*rbuffer[4];
            _Pi[m-myid*nx1+gz_offset] = 0.5*rbuffer[2]+0.5*rbuffer[5];
          }
        }

      }
    }
    // }}}
//#endif
//#if 0
    // take two steps of the finer mesh
    for (k=0;k<2;k++) {
     // Finer mesh evolution {{{
     // ------------------------------- iter 1
     // communicate because of filling ghostzones
     communicate(numprocs,myid,_r,_chi);
     communicate(numprocs,myid,_r,_Phi);
     communicate(numprocs,myid,_r,_Pi);
     calc_rhs(numprocs,myid,_r,_chi,_Phi,_Pi,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);


     for (j=_lower;j<_upper;j++) {
       _chi_np1[j] = _chi[j] + _rhs_chi[j]*dt1; 
       _Phi_np1[j] = _Phi[j] + _rhs_Phi[j]*dt1; 
       _Pi_np1[j] =  _Pi[j] + _rhs_Pi[j]*dt1; 

     }

     // r = 0 boundary
     if ( myid == 0 ) {
       _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
       _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
       _Phi_np1[1] = 0.5*_Phi_np1[2];
     }

     //---------------------------------- iter 2
     communicate(numprocs,myid,_r,_chi_np1);
     communicate(numprocs,myid,_r,_Phi_np1);
     communicate(numprocs,myid,_r,_Pi_np1);
     calc_rhs(numprocs,myid,_r,_chi_np1,_Phi_np1,_Pi_np1,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);

     for (j=_lower;j<_upper;j++) {
       _chi_np1[j] = 0.75*_chi[j] + 0.25*_chi_np1[j] + 0.25*_rhs_chi[j]*dt1; 
       _Phi_np1[j] = 0.75*_Phi[j] + 0.25*_Phi_np1[j] + 0.25*_rhs_Phi[j]*dt1; 
       _Pi_np1[j]  = 0.75*_Pi[j]  + 0.25*_Pi_np1[j]  + 0.25*_rhs_Pi[j]*dt1; 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
       _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
       _Phi_np1[1] = 0.5*_Phi_np1[2];
     }

     //---------------------------------- iter 3
     communicate(numprocs,myid,_r,_chi_np1);
     communicate(numprocs,myid,_r,_Phi_np1);
     communicate(numprocs,myid,_r,_Pi_np1);
     calc_rhs(numprocs,myid,_r,_chi_np1,_Phi_np1,_Pi_np1,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);

     for (j=_lower;j<_upper;j++) {
       _chi_np1[j] = 1./3*_chi[j] + 2./3*(_chi_np1[j] + _rhs_chi[j]*dt1); 
       _Phi_np1[j] = 1./3*_Phi[j] + 2./3*(_Phi_np1[j] + _rhs_Phi[j]*dt1); 
       _Pi_np1[j]  = 1./3*_Pi[j]  + 2./3*(_Pi_np1[j]  + _rhs_Pi[j]*dt1); 
     }

     // r = 0 boundary
     if ( myid == 0 ) {
       _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
       _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
       _Phi_np1[1] = 0.5*_Phi_np1[2];
     }

     // }}}

      _chi.swap(_chi_np1);
      _Phi.swap(_Phi_np1);
      _Pi.swap(_Pi_np1);

      communicate(numprocs,myid,_r,_chi);
      communicate(numprocs,myid,_r,_Phi);
      communicate(numprocs,myid,_r,_Pi);

      sprintf(basename,"chi1");
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,dt*i+(k+1)*dt1,shape1,cnames,1,&*_r.begin(),&*_chi.begin());
      sprintf(basename,"Phi1");
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,dt*i+(k+1)*dt1,shape1,cnames,1,&*_r.begin(),&*_Phi.begin());
      sprintf(basename,"Pi1");
      sprintf(filename,"%d%s",myid,basename);
      gft_out_full(filename,dt*i+(k+1)*dt1,shape1,cnames,1,&*_r.begin(),&*_Pi.begin());
    }
//#endif

    chi.swap(chi_np1);
    Phi.swap(Phi_np1);
    Pi.swap(Pi_np1);

//#if 0
    // injecting -- overwrite coarse mesh points with finer mesh result {{{
    count = 0;
    for (j=0;j<global_nx1-10;j = j+2) {
      if ( restrict_coarse[count] == -1 || restrict_fine[j] == -1 ) {
        std::cerr << " Problem " << std::endl;
        exit(0);
      }

      if ( restrict_coarse[count] == restrict_fine[j] && myid == restrict_fine[j] ) {
        // no communication needed for restriction
         chi[count-myid*nx0+gz_offset] = _chi[j-myid*nx1+gz_offset];
         Phi[count-myid*nx0+gz_offset] = _Phi[j-myid*nx1+gz_offset];
         Pi[count-myid*nx0+gz_offset] = _Pi[j-myid*nx1+gz_offset];
      } else {
        if ( myid == restrict_coarse[count] ) {
           MPI_Recv(rbuffer,3,MPI_DOUBLE,restrict_fine[j],tag2,MPI_COMM_WORLD,&status);
           chi[count-myid*nx0+gz_offset] = rbuffer[0];
           Phi[count-myid*nx0+gz_offset] = rbuffer[1];
           Pi[count-myid*nx0+gz_offset] = rbuffer[2];
        } else if ( myid == restrict_fine[j] ) {
           buffer[0] = _chi[j-myid*nx1+gz_offset];
           buffer[1] = _Phi[j-myid*nx1+gz_offset];
           buffer[2] = _Pi[j-myid*nx1+gz_offset];
           MPI_Send(buffer,3,MPI_DOUBLE,restrict_coarse[count],tag2,MPI_COMM_WORLD);
        }
      }
      count++;
    } 
    // }}}
//#endif

    communicate(numprocs,myid,r,chi);
    communicate(numprocs,myid,r,Phi);
    communicate(numprocs,myid,r,Pi);

    sprintf(basename,"chi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt*(i+1),shape,cnames,1,&*r.begin(),&*chi.begin());
    sprintf(basename,"Phi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt*(i+1),shape,cnames,1,&*r.begin(),&*Phi.begin());
    sprintf(basename,"Pi0");
    sprintf(filename,"%d%s",myid,basename);
    gft_out_full(filename,dt*(i+1),shape,cnames,1,&*r.begin(),&*Pi.begin());
  }
  t2 = MPI_Wtime();
  if ( myid == 0 ) {
    std::cout << " Elapsed time: " << t2-t1 << std::endl;
  }

  MPI_Finalize();
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
