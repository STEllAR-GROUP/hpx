//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "particle.hpp"

#include <boost/lexical_cast.hpp>

#include <string>
#include <sstream>
#include <fstream>

double unifRand() {
  return rand()/double(RAND_MAX);
}
double unifRand(double a,double b)
{
  return (b-a)*unifRand() + a;
}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void particle::init(std::size_t objectid,parameter const& par)
    {
        // 
        srand(objectid+5);

        // initial data

        // initial mesh
        double pi = 4.0*atan(1.0);
        std::size_t toroidal_domain_location=objectid/par->npartdom;
        double tmp1 = (double) toroidal_domain_location;
        double tmp2 = (double) par->ntoroidal;
        double tmp3 = (double) (toroidal_domain_location+1);
        zetamin_ = 2.0*pi*tmp1/tmp2;
        zetamax_ = 2.0*pi*tmp3/tmp2;

        mtheta_.resize(par->mpsi+1);
        // --- Define poloidal grid ---
        double tmp5 = (double) par->mpsi;
        deltar_ = (par->a1-par->a0)/tmp5;

        // grid shift associated with fieldline following coordinates
        double tmp6 = (double) par->mthetamax;
        double tdum = 2.0*pi*par->a1/tmp6;
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          std::size_t two = 2;
          double tmp7 = pi*r/tdum + 0.5;
          std::size_t tmp8 = (std::size_t) tmp7;
          mtheta_[i] = std::max(two,std::min(par->mthetamax,two*tmp8)); // even # poloidal grid
        }

        // number of grids on a poloidal plane
        mgrid_ = 0;
        for (std::size_t i=0;i<mtheta_.size();i++) {
          mgrid_ += mtheta_[i] + 1;
        }
        mzeta_ = par->mzetamax/par->ntoroidal;
        std::size_t mi = par->micell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of ions per processor

        double rmi = 1.0/(mi*par->npartdom);
        double pi2_inv = 0.5/pi;
        double delr = 1.0/deltar_;
        double ainv = 1.0/par->a;
        double w_initial = 1.0e-3;
        if ( par->nonlinear < 0.5 ) w_initial = 1.0e-12;
        std::size_t ntracer = 0;
        if ( objectid == 0 ) ntracer = 1;
       
        std::size_t nparam;
        if ( par->track_particles ) {
          // Not implemented yet
          nparam = 7;
        } else {
          // No tagging of the particles
          nparam = 6;
        }
        double tmp13 = (double) mi;
        std::size_t mimax = mi + 100*std::ceil(sqrt(tmp13)); // ions array upper bound

        zion_.resize(nparam+1,mimax+1,1);
        zion0_.resize(nparam+1,mimax+1,1);
        for (std::size_t m=1;m<=mi;m++) {
          zion_(1,m,0) = sqrt(par->a0*par->a0 + ( (m+objectid*mi)-0.5 )*(par->a1*par->a1-par->a0*par->a0)*rmi);
        }

        if ( par->track_particles ) BOOST_ASSERT(false); // not implemented yet

        // Set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
        for (std::size_t i=2;i<=6;i++) {
          for (std::size_t j=1;j<=mi;j++) {
            zion_(i,j,0) = unifRand(0,1);
          }
        }

        // poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
        for (std::size_t m=1;m<=mi;m++) {
          zion_(2,m,0) = 2.0*pi*(zion_(2,m,0)-0.5);
          zion0_(2,m,0) = zion_(2,m,0); // zion0(2,:) for temporary storage
        }

        for (std::size_t i=1;i<=10;i++) {
          for (std::size_t m=1;m<=mi;m++) {
            zion_(2,m,0) = zion_(2,m,0)*pi2_inv+10.0; // period of 1
            zion_(2,m,0) = 2.0*pi*(zion_(2,m,0)-floor(zion_(2,m,0)));
          }
        }

        // Maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
        double SMALL = 1.e-20;
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        for (std::size_t m=1;m<=mi;m++) {
          double z4tmp = zion_(4,m,0);
          zion_(4,m,0) = zion_(4,m,0)-0.5;
          if ( zion_(4,m,0) > 0.0 ) zion0_(4,m,0) = 1.0; 
          else zion0_(4,m,0) = -1.0;
          zion_(4,m,0) = sqrt( std::max(SMALL,log(1.0/std::max(SMALL,pow(zion_(4,m,0),2)))));
          zion_(4,m,0) = zion_(4,m,0) - (c0+c1*zion_(4,m,0)+c2*pow(zion_(4,m,0),2))/
                                      (1.0+d1*zion_(4,m,0)+d2*pow(zion_(4,m,0),2)+d3*pow(zion_(4,m,0),3));
          if ( zion_(4,m,0) > par->umax ) zion_(4,m,0) = z4tmp;
        }

        for (std::size_t m=1;m<=mi;m++) {
          // toroidal:  uniformly distributed in zeta
          zion_(3,m,0) = zetamin_+(zetamax_-zetamin_)*zion_(3,m,0);
          zion_(4,m,0) = zion0_(4,m,0)*std::min(par->umax,zion_(4,m,0));

          // initial random weight
          zion_(5,m,0) = 2.0*w_initial*(zion_(5,m,0)-0.5)*(1.0+cos(zion_(2,m,0)));

          // Maxwellian distribution in v_perp, <v_perp^2>=1.0
          zion_(6,m,0) = std::max(SMALL,std::min(par->umax*par->umax,-log(std::max(SMALL,zion_(6,m,0)))));
        }

        // transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
        double vthi = par->gyroradius*abs(par->qion)/par->aion;
        for (std::size_t m=1;m<=mi;m++) {
          zion0_(1,m,0) = 1.0/(1.0+zion_(1,m,0)*cos(zion_(2,m,0))); // B-field
          zion_(1,m,0) = 0.5*zion_(1,m,0)*zion_(1,m,0);
          zion_(4,m,0) = vthi*zion_(4,m,0)*par->aion/(par->qion*zion0_(1,m,0));
          zion_(6,m,0) = sqrt(par->aion*vthi*vthi*zion_(6,m,0)/zion0_(1,m,0));
        }
        
        if ( objectid == 0 ) {
          zion_(1,ntracer,0) = 0.5*pow((0.5*(par->a0+par->a1)),2);
          zion_(2,ntracer,0) = 0.0;
          zion_(3,ntracer,0) = 0.5*(zetamin_+zetamax_);
          zion_(4,ntracer,0) = 0.5*vthi*par->aion/par->qion;
          zion_(5,ntracer,0) = 0.0;
          zion_(6,ntracer,0) = sqrt(par->aion*vthi*vthi);
        }

        if ( par->iload == false ) {
          for (std::size_t m=1;m<=mi;m++) {
            zion0_(6,m,0) = 1.0;
          }
        } else {
          std::cerr << " Not implemented yet " << std::endl;
        }

        if ( par->nhybrid > 0 ) {
          std::cerr << " Not implemented yet " << std::endl;
        }
    }
}}


