
#include <iostream>
#include <vector>
#include <math.h>
#include <sdf.h>
using namespace std;

int initial_data(std::vector<double> &r,
                 std::vector<double> &chi,
                 std::vector<double> &Phi,
                 std::vector<double> &Pi,
                 std::vector<double> &energy,int,double,int);

int calc_rhs(std::vector<double> &r,
             std::vector<double> &chi,
             std::vector<double> &Phi,
             std::vector<double> &Pi,
             std::vector<double> &rhs_chi,
             std::vector<double> &rhs_Phi,
             std::vector<double> &rhs_Pi,int PP,double eps); 

int main() {

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
  //   gw==8 results in noise
  int gw  = 10;
  int nx0 = 40;
  // CODiss 
  double eps = 0.0;

  int nx1 = 19;
  if ( nx1%2 != 0 ) {
    std::cerr << " PROBLEM : nx1 needs to be odd " << std::endl;
  }
  int nt0 = 400;
  double time = 0.0;
  int shape[3];
  int shape0[3];
  int PP = 7;

  double dx = 15.0/(nx0-1);
  double dt = 0.15*dx;
  double dt2 = 0.5*dt;

  initial_data(r,chi,Phi,Pi,energy,nx0,dx,PP);

  initial_data(_r,_chi,_Phi,_Pi,_energy,nx1,0.5*dx,PP);

  cout<<"Coarse--dx/nx/maxx:"<<dx<<" "<<nx0<<" "<<r[nx0-1]<<"\n";
  cout<<"      "<<(nx0-1)*1.0*dx<<"\n";
  cout<<"Fine  --dx/nx/maxx:"<<0.5*dx<<" "<<nx1<<" "<<_r[nx1-1]<<"\n";
  cout<<"      "<<(nx1-1)*0.5*dx<<"\n";

  chi_np1.resize(chi.size());
  Phi_np1.resize(chi.size());
  Pi_np1.resize(chi.size());

  rhs_chi.resize(chi.size());
  rhs_Phi.resize(chi.size());
  rhs_Pi.resize(chi.size());

  _chi_np1.resize(_r.size());
  _Phi_np1.resize(_r.size());
  _Pi_np1.resize(_r.size());

  _rhs_chi.resize(_r.size());
  _rhs_Phi.resize(_r.size());
  _rhs_Pi.resize(_r.size());

  shape[0]  = nx0;

  // This is just so that the number of points output on level 1 
  // coincides with an output quirk of parallex at t = 0
  shape0[0] = nx1;
  sprintf(cnames,"r");
  gft_out_full("r",time,shape,cnames,1,&*r.begin(),&*r.begin());
  gft_out_full("chi",time,shape,cnames,1,&*r.begin(),&*chi.begin());
  gft_out_full("Phi",time,shape,cnames,1,&*r.begin(),&*Phi.begin());
  gft_out_full("Pi",time,shape,cnames,1,&*r.begin(),&*Pi.begin());
  //gft_out_full("energy",time,shape,cnames,1,&*r.begin(),&*energy.begin());

  gft_out_full("r_1",time,shape0,cnames,1,&*_r.begin(),&*_r.begin());
  gft_out_full("chi_1",time,shape0,cnames,1,&*_r.begin(),&*_chi.begin());
  gft_out_full("Phi_1",time,shape0,cnames,1,&*_r.begin(),&*_Phi.begin());
  gft_out_full("Pi_1",time,shape0,cnames,1,&*_r.begin(),&*_Pi.begin());
  //gft_out_full("energy_1",time,shape0,cnames,1,&*_r.begin(),&*_energy.begin());

  for (i=0;i<nt0;i++) {
     // Coarse mesh evolution {{{
     // ------------------------------- iter 1
     calc_rhs(r,chi,Phi,Pi,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     // r = 0 boundary
     //chi_np1[0] = 4./3*chi[1] -1./3*chi[2];
     //Pi_np1[0]  = 4./3*Pi[1]  -1./3*Pi[2];

     for (j=1;j<chi.size();j++) {
       chi_np1[j] = chi[j] + rhs_chi[j]*dt; 
       Phi_np1[j] = Phi[j] + rhs_Phi[j]*dt; 
       Pi_np1[j] =  Pi[j] + rhs_Pi[j]*dt; 
     }

     // r = 0 boundary
     chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
     Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
     Phi_np1[1] = 0.5*Phi_np1[2];

     //---------------------------------- iter 2
     calc_rhs(r,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     // r = 0 boundary
     //chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
     //Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];

     for (j=1;j<chi.size();j++) {
       chi_np1[j] = 0.75*chi[j] + 0.25*chi_np1[j] + 0.25*rhs_chi[j]*dt; 
       Phi_np1[j] = 0.75*Phi[j] + 0.25*Phi_np1[j] + 0.25*rhs_Phi[j]*dt; 
       Pi_np1[j]  = 0.75*Pi[j]  + 0.25*Pi_np1[j]  + 0.25*rhs_Pi[j]*dt; 
     }

     // r = 0 boundary
     chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
     Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
     Phi_np1[1] = 0.5*Phi_np1[2];
 
     //---------------------------------- iter 3
     calc_rhs(r,chi_np1,Phi_np1,Pi_np1,rhs_chi,rhs_Phi,rhs_Pi,PP,eps);

     // r = 0 boundary
     //chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
     //Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];

     for (j=1;j<chi.size();j++) {
       chi_np1[j] = 1./3*chi[j] + 2./3*(chi_np1[j] + rhs_chi[j]*dt); 
       Phi_np1[j] = 1./3*Phi[j] + 2./3*(Phi_np1[j] + rhs_Phi[j]*dt); 
       Pi_np1[j]  = 1./3*Pi[j]  + 2./3*(Pi_np1[j]  + rhs_Pi[j]*dt); 

     }

     // r = 0 boundary
     chi_np1[0] = 4./3*chi_np1[1] -1./3*chi_np1[2];
     Pi_np1[0]  = 4./3*Pi_np1[1]  -1./3*Pi_np1[2];
     Phi_np1[1] = 0.5*Phi_np1[2];

     for (j=0;j<energy.size();j++) {
       energy[j] = 0.5*r[j]*r[j]*(Pi_np1[j]*Pi_np1[j] + Phi_np1[j]*Phi_np1[j]) 
                                    - r[j]*r[j]*pow(chi_np1[j],PP+1.0)/(PP+1.0);
     }
    

     cout<<"Output: coarse level at the advanced time: chi_np1\n";
     gft_out_full("chi_np1",dt*(i+1),shape,cnames,1,&*r.begin(),&*chi_np1.begin());
     gft_out_full("Phi_np1",dt*(i+1),shape,cnames,1,&*r.begin(),&*Phi_np1.begin());
     gft_out_full("Pi_np1",dt*(i+1),shape,cnames,1,&*r.begin(),&*Pi_np1.begin());
     //gft_out_full("energy_np1",dt*(i+1),shape,cnames,1,&*r.begin(),&*energy.begin());
     // }}}

    if ( i != 0 && 1==1) {
      // fill the last 8 points in the finer mesh using the coarse mesh {{{
       cout<<"Resetting boundaries as per taper....gw="<<gw<<"\n";
       for (j=0;j<gw/2;j++) {
          int m,mm;
          m  = nx1-2*j;
          mm = (m-1)/2;
          cout<<" "<<m<<" "<<mm<<" ";
         _chi[m-1] = chi[mm]; 
         _Phi[m-1] = Phi[mm]; 
          _Pi[m-1] =  Pi[mm]; 
         _chi[m] = 0.5*chi[mm]+0.5*chi[mm+1]; 
         _Phi[m] = 0.5*Phi[mm]+0.5*Phi[mm+1]; 
          _Pi[m] = 0.5* Pi[mm]+0.5* Pi[mm+1]; 
      }
          cout<<" Done.\n";

     } else {  cout<<"*NOT* Resetting boundaries as per taper....\n";
    // }}}
    }
     cout<<"Output: coarse level ID at the advanced time: chi_n\n";
     gft_out_full("chi_n",dt*(i+0),shape,cnames,1,&*r.begin(),&*chi.begin());
     gft_out_full("Phi_n",dt*(i+0),shape,cnames,1,&*r.begin(),&*Phi.begin());
     gft_out_full("Pi_n",dt*(i+0),shape,cnames,1,&*r.begin(),&*Pi.begin());
     cout<<"Output: fine   level ID at the advanced time: chi_n\n";
     gft_out_full("chi_n",dt*i,shape0,cnames,1,&*_r.begin(),&*_chi.begin());
     gft_out_full("Phi_n",dt*i,shape0,cnames,1,&*_r.begin(),&*_Phi.begin());
     gft_out_full("Pi_n",dt*i,shape0,cnames,1,&*_r.begin(),&*_Pi.begin());
     cout<<"....these two should agree with each other in taper\n";

    // take two steps of the finer mesh
    for (k=0;k<2;k++) {
     // Fine mesh evolution {{{
     // ------------------------------- iter 1
     calc_rhs(_r,_chi,_Phi,_Pi,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);

     // r = 0 boundary
     //_chi_np1[0] = 4./3*_chi[1] -1./3*_chi[2];
     //_Pi_np1[0]  = 4./3*_Pi[1]  -1./3*_Pi[2];

     for (j=1;j<_chi.size();j++) {
       _chi_np1[j] = _chi[j] + _rhs_chi[j]*dt2; 
       _Phi_np1[j] = _Phi[j] + _rhs_Phi[j]*dt2; 
       _Pi_np1[j] =  _Pi[j] + _rhs_Pi[j]*dt2; 
     }

     // r = 0 boundary
     _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
     _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
     _Phi_np1[1] = 0.5*_Phi_np1[2];

     gft_out_full("fchi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_chi_np1.begin());
     gft_out_full("fPhi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Phi_np1.begin());
     gft_out_full("fPi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Pi_np1.begin());

     //---------------------------------- iter 2
     calc_rhs(_r,_chi_np1,_Phi_np1,_Pi_np1,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);

     // r = 0 boundary
     //_chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
     //_Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];

     for (j=1;j<_chi.size();j++) {
       _chi_np1[j] = 0.75*_chi[j] + 0.25*_chi_np1[j] + 0.25*_rhs_chi[j]*dt2; 
       _Phi_np1[j] = 0.75*_Phi[j] + 0.25*_Phi_np1[j] + 0.25*_rhs_Phi[j]*dt2; 
       _Pi_np1[j]  = 0.75*_Pi[j]  + 0.25*_Pi_np1[j]  + 0.25*_rhs_Pi[j]*dt2; 
     }

     // r = 0 boundary
     _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
     _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
     _Phi_np1[1] = 0.5*_Phi_np1[2];

     gft_out_full("fchi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_chi_np1.begin());
     gft_out_full("fPhi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Phi_np1.begin());
     gft_out_full("fPi_np1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Pi_np1.begin());

     //---------------------------------- iter 3
     calc_rhs(_r,_chi_np1,_Phi_np1,_Pi_np1,_rhs_chi,_rhs_Phi,_rhs_Pi,PP,eps);

     // r = 0 boundary
     //_chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
     //_Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];

     for (j=1;j<_chi.size();j++) {
       _chi_np1[j] = 1./3*_chi[j] + 2./3*(_chi_np1[j] + _rhs_chi[j]*dt2); 
       _Phi_np1[j] = 1./3*_Phi[j] + 2./3*(_Phi_np1[j] + _rhs_Phi[j]*dt2); 
       _Pi_np1[j]  = 1./3*_Pi[j]  + 2./3*(_Pi_np1[j]  + _rhs_Pi[j]*dt2); 

     }

     // r = 0 boundary
     _chi_np1[0] = 4./3*_chi_np1[1] -1./3*_chi_np1[2];
     _Pi_np1[0]  = 4./3*_Pi_np1[1]  -1./3*_Pi_np1[2];
     _Phi_np1[1] = 0.5*_Phi_np1[2];

     for (j=0;j<_energy.size();j++) {
       _energy[j] = 0.5*_r[j]*_r[j]*(_Pi_np1[j]*_Pi_np1[j] + _Phi_np1[j]*_Phi_np1[j]) 
                                    - _r[j]*_r[j]*pow(_chi_np1[j],PP+1.0)/(PP+1.0);
     }
    
     gft_out_full("chi_1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_chi_np1.begin());
     gft_out_full("Phi_1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Phi_np1.begin());
     gft_out_full("Pi_1",dt*i+dt2*(k+1),shape0,cnames,1,&*_r.begin(),&*_Pi_np1.begin());
     // }}}

      _chi.swap(_chi_np1);
      _Phi.swap(_Phi_np1);
      _Pi.swap(_Pi_np1);
    }

    chi.swap(chi_np1);
    Phi.swap(Phi_np1);
    Pi.swap(Pi_np1);

    
    if (1==1) {
       cout<<"Injecting...\n";
       //  INJECTION:
       //  INJECTION:
       //  INJECTION:
       //  INJECTION:
       // Overwrite coarse mesh points with finer mesh result
       count = 0;
       for (j=0;j<nx1-10;j = j+2) {
         chi[count] = _chi[j];
         Phi[count] = _Phi[j];
         Pi[count]  = _Pi[j];
         count++;
       } 
    } else {
       cout<<"Not injecting\n";
    } 

    //  OUTPUT:
    //  OUTPUT:
    //  OUTPUT:
    //  OUTPUT:
    gft_out_full("chi",dt*(i+1),shape,cnames,1,&*r.begin(),&*chi.begin());
    gft_out_full("Phi",dt*(i+1),shape,cnames,1,&*r.begin(),&*Phi.begin());
    gft_out_full("Pi",dt*(i+1),shape,cnames,1,&*r.begin(),&*Pi.begin());
    gft_out_full("energy",dt*(i+1),shape,cnames,1,&*r.begin(),&*energy.begin());
  }

  return 0;
}

//
//
//
//
//
int initial_data(std::vector<double> &r,
                 std::vector<double> &chi,
                 std::vector<double> &Phi,
                 std::vector<double> &Pi,
                 std::vector<double> &energy,int nx0,double dx,int PP) {


  int i;
  for (i=0;i<nx0;i++) {
    r.push_back(i*dx);
  }

  double amp = 0.01;
  double delta = 1.0;
  double R0 = 8.0;

  for (i=0;i<nx0;i++) {
    chi.push_back(amp*exp(-(r[i]-R0)*(r[i]-R0)/(delta*delta)));  
    Phi.push_back(amp*exp(-(r[i]-R0)*(r[i]-R0)/(delta*delta)) * ( -2.*(r[i]-R0)/(delta*delta)  )   );
    Pi.push_back(0.0);
    energy.push_back( 0.5*r[i]*r[i]*(Pi[i]*Pi[i] + Phi[i]*Phi[i])-r[i]*r[i]*pow(chi[i],PP+1)/(PP+1) );
  }

  return 0;
}

int calc_rhs(std::vector<double> &r,
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

  for (i=1;i<r.size()-1;i++) {
    rhs_chi[i] = Pi[i] + eps*diss_chi[i];
    rhs_Phi[i] = 1./(2.*dr) * ( Pi[i+1] - Pi[i-1] ) + eps*diss_Phi[i];
    rhs_Pi[i]  = 3 * ( r[i+1]*r[i+1]*Phi[i+1] - r[i-1]*r[i-1]*Phi[i-1] )/( pow(r[i+1],3) - pow(r[i-1],3) )
                    + 0.0*pow(chi[i],PP) + eps*diss_Pi[i];

  }

  //std::cout << " TEST right boundary " << r.size()-1 << std::endl;
  i = r.size()-1;
  rhs_chi[i] = Pi[i];
  rhs_Phi[i] = -(3.*Phi[i] - 4.*Phi[i-1] + Phi[i-2])/(2.*dr) - Phi[i]/r[i];
  rhs_Pi[i]  = -Pi[i]/r[i] - (3.*Pi[i] - 4.*Pi[i-1] + Pi[i-2])/(2.*dr);

  return 0;
}
