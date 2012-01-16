//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/lexical_cast.hpp>

#include "point.hpp"
#include "../stubs/point.hpp"

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
    void point::init(std::size_t objectid,parameter const& par)
    {
        idx_ = objectid;

        tauii_ = -1.0; // initially collisionless

        // initial mesh
        particle_domain_location_=objectid%par->npartdom;
        toroidal_domain_location_=objectid/par->npartdom;
        pi_ = 4.0*atan(1.0);
   
        // Domain decomposition in toroidal direction
        mzeta_ = par->mzetamax/par->ntoroidal;
        double tmp1 = (double) toroidal_domain_location_;
        double tmp2 = (double) par->ntoroidal;
        double tmp3 = (double) (toroidal_domain_location_+1);
        zetamin_ = 2.0*pi_*tmp1/tmp2;
        zetamax_ = 2.0*pi_*tmp3/tmp2;

        double tmp4 = (double) mzeta_;
        deltaz_ = (zetamax_-zetamin_)/tmp4;

        myrank_toroidal_ = objectid;
        left_pe_ = (myrank_toroidal_-1+par->ntoroidal)% par->ntoroidal;
        right_pe_ = (myrank_toroidal_+1)% par->ntoroidal;

        qtinv_.resize(par->mpsi+1);
        itran_.resize(par->mpsi+1);
        mtheta_.resize(par->mpsi+1);
        deltat_.resize(par->mpsi+1);
        rtemi_.resize(par->mpsi+1);
        rteme_.resize(par->mpsi+1);
        rden_.resize(par->mpsi+1);
        igrid_.resize(par->mpsi+1);
        pmarki_.resize(par->mpsi+1);
        pmarke_.resize(par->mpsi+1);
        phi00_.resize(par->mpsi+1);
        phip00_.resize(par->mpsi+1);
        hfluxpsi_.resize(par->mpsi+1);
        zonali_.resize(par->mpsi+1);
        zonale_.resize(par->mpsi+1);
        gradt_.resize(par->mpsi+1);
        eigenmode_.resize(par->m_poloidal,par->num_mode,par->mpsi);

        // --- Define poloidal grid ---
        double tmp5 = (double) par->mpsi;
        deltar_ = (par->a1-par->a0)/tmp5;

        // grid shift associated with fieldline following coordinates
        double tmp6 = (double) par->mthetamax;
        double tdum = 2.0*pi_*par->a1/tmp6;

        // initial data
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          std::size_t two = 2;
          double tmp7 = pi_*r/tdum + 0.5;
          std::size_t tmp8 = (std::size_t) tmp7;
          mtheta_[i] = (std::max)(two,(std::min)(par->mthetamax,two*tmp8)); // even # poloidal grid
          deltat_[i] = 2.0*pi_/mtheta_[i];
          double q = par->q0 + par->q1*r/par->a + par->q2*r*r/(par->a*par->a);
          double tmp9 = mtheta_[i]/q + 0.5;
          std::size_t tmp10 = (std::size_t) tmp9;
          itran_[i] = tmp10;
          double tmp11 = (double) mtheta_[i];
          double tmp12 = (double) itran_[i];
          qtinv_[i] = tmp11/tmp12;  // q value for coordinate transformation
          qtinv_[i] = 1.0/qtinv_[i]; // inverse q to avoid divide operation
          itran_[i] = itran_[i] - mtheta_[i]*(itran_[i]/mtheta_[i]);
        }

        // When doing mode diagnostics, we need to switch from the field-line following
        // coordinates alpha-zeta to a normal geometric grid in theta-zeta. This
        // translates to a greater number of grid points in the zeta direction, which
        // is mtdiag. Precisely, mtdiag should be mtheta/q but since mtheta changes
        // from one flux surface to another, we use a formula that gives us enough
        // grid points for all the flux surfaces considered.
        mtdiag_ = (par->mthetamax/par->mzetamax)*par->mzetamax;
        if ( par->nonlinear > 0.5 ) mtdiag_ = mtdiag_/2;

        // starting point for a poloidal grid
        igrid_[0] = 1;
        for (std::size_t i=1;i<par->mpsi+1;i++) {
          igrid_[i] = igrid_[i-1]+mtheta_[i-1]+1;
        }

        // number of grids on a poloidal plane
        mgrid_ = 0;
        for (std::size_t i=0;i<mtheta_.size();i++) {
          mgrid_ += mtheta_[i] + 1;
        }
        std::size_t mi_local = par->micell*(mgrid_-par->mpsi)*mzeta_;  // # of ions in toroidal domain
        mi_ = par->micell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of ions per processor
        if ( mi_ <  (mi_local%par->npartdom) ) mi_++;
        std::size_t me_local = par->mecell*(mgrid_-par->mpsi)*mzeta_;  // # of electrons in toroidal domain
        std::size_t me_ = par->mecell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of electrons per processor
        if ( me_ < (me_local%par->npartdom) ) me_++;

        double tmp13 = (double) mi_;
        double tmp14 = (double) me_;
        std::size_t mimax = static_cast<std::size_t>(mi_ + 100*std::ceil(sqrt(tmp13))); // ions array upper bound
        std::size_t memax = static_cast<std::size_t>(me_ + 100*std::ceil(sqrt(tmp14))); // electrons array upper bound

        pgyro_.resize(5,mgrid_+1,1);
        tgyro_.resize(5,mgrid_+1,1);
        markeri_.resize(mzeta_+1,mgrid_+1,1);
        densityi_.resize(mzeta_+1,mgrid_+1,1);
        phi_.resize(mzeta_+1,mgrid_+1,1);
        evector_.resize(3,mzeta_+1,mgrid_);
        jtp1_.resize(3,mgrid_,mzeta_+1);
        jtp2_.resize(3,mgrid_,mzeta_+1);
        wtp1_.resize(3,mgrid_,mzeta_+1);
        wtp2_.resize(3,mgrid_,mzeta_+1);
        dtemper_.resize(mgrid_,mzeta_,1);
        heatflux_.resize(mgrid_,mzeta_,1);

        pfluxpsi_.resize(par->mflux);
        rdtemi_.resize(par->mflux);
        rdteme_.resize(par->mflux);

        // initialize arrays
        // temperature and density on the grid, T_i=n_0=1 at mid-radius
        std::fill( rtemi_.begin(),rtemi_.end(),1.0);
        std::fill( rteme_.begin(),rteme_.end(),1.0);
        std::fill( rden_.begin(),rden_.end(),1.0);
        std::fill( phip00_.begin(),phip00_.end(),0.0);
        std::fill( zonali_.begin(),zonali_.end(),0.0);
        std::fill( zonale_.begin(),zonale_.end(),0.0);
        for (std::size_t i=0;i<phi_.size();i++) {
          phi_[i] = 0.0;
        }
        std::fill( pfluxpsi_.begin(),pfluxpsi_.end(),0.0);
        std::fill( rdtemi_.begin(),rdtemi_.end(),0.0);
        std::fill( rdteme_.begin(),rdteme_.end(),0.0);

        // # of marker per grid, Jacobian=(1.0+r*cos(theta+r*sin(theta)))*(1.0+r*cos(theta))
        std::fill( pmarki_.begin(),pmarki_.end(),0.0);

        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          for (std::size_t j=1;j<=mtheta_[i];j++) {
            std::size_t ij = igrid_[i] + j;
            for (std::size_t k=1;k<=mzeta_;k++) {
              double zdum = zetamin_ + k*deltaz_;
              double tdum = j*deltat_[i]+zdum*qtinv_[i];
              markeri_(k,ij,0) = pow(1.0+r*cos(tdum),2);
              pmarki_[i] = pmarki_[i] + markeri_(k,ij,0);
            }
          }
          double rmax = (std::min)(par->a1,r+0.5*deltar_);
          double rmin = (std::max)(par->a0,r-0.5*deltar_);
          double tmp15 = (double) mi_*par->npartdom;
          tdum = tmp15*(rmax*rmax-rmin*rmin)/(par->a1*par->a1-par->a0*par->a0);
          for (std::size_t j=1;j<=mtheta_[i];j++) {
            std::size_t ij = igrid_[i] + j;
            for (std::size_t k=1;k<=mzeta_;k++) {
              markeri_(k,ij,0) = tdum*markeri_(k,ij,0)/pmarki_[i];
              markeri_(k,ij,0) = 1.0/markeri_(k,ij,0);
            }
          }
          pmarki_[i] = 1.0/(par->ntoroidal*tdum);
          for (std::size_t j=0;j<markeri_.isize();j++) {
            markeri_(j,igrid_[i],0) = markeri_(j,igrid_[i]+mtheta_[i],0);
          }
        }
        std::size_t nparam;
        if ( par->track_particles ) {
          // Not implemented yet
          nparam = 7;
        } else {
          // No tagging of the particles
          nparam = 6;
        }
        zion_.resize(nparam+1,mimax+1,1);
        zion0_.resize(nparam+1,mimax+1,1);
        jtion0_.resize(nparam+1,mimax+1,1);
        jtion1_.resize(5,mimax+1,1);
        kzion_.resize(mimax+1);
        wzion_.resize(mimax+1);
        wpion_.resize(5,mimax+1,1);
        wtion0_.resize(5,mimax+1,1);
        wtion1_.resize(5,mimax+1,1);

        if ( par->nhybrid > 0 ) {
          BOOST_ASSERT(false);
          // not implemented yet
        }

        // 4-point gyro-averaging for sqrt(mu)=gyroradius on grid of magnetic coordinates
        // rho=gyroradius*sqrt(2/(b/b_0))*sqrt(mu/mu_0), mu_0*b_0=m*v_th^2
        // dtheta/delta_x=1/(r*(1+r*cos(theta))), delta_x=poloidal length increase
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          for (std::size_t j=0;j<=mtheta_[i];j++) {
            std::size_t ij = igrid_[i] + j;
            double tdum = deltat_[i]*j;
            double b = 1.0/(1.0+r*cos(tdum));
            double dtheta_dx = 1.0/r;
            // first two points perpendicular to field line on poloidal surface
            double rhoi = sqrt(2.0/b)*par->gyroradius;

            pgyro_(1,ij,0) = -rhoi;
            pgyro_(2,ij,0) = rhoi;
            // non-orthorgonality between psi and theta: tgyro=-rhoi*dtheta_dx*r*sin(tdum)
            tgyro_(1,ij,0) = 0.0;
            tgyro_(2,ij,0) = 0.0;

            // the other two points tangential to field line
            tgyro_(3,ij,0) = -rhoi*dtheta_dx;
            tgyro_(4,ij,0) = rhoi*dtheta_dx;
            pgyro_(3,ij,0) = rhoi*0.5*rhoi/r;
            pgyro_(4,ij,0) = rhoi*0.5*rhoi/r;
          }
        }

        // initiate radial interpolation for grid
        for (std::size_t k=1;k<=mzeta_;k++) {
          double zdum = zetamin_ + k*deltaz_;
          for (std::size_t i=1;i<par->mpsi;i++) {
            for (std::size_t ip=1;ip<=2;ip++) {
              std::size_t indp = (std::min)(par->mpsi,i+ip);
              double tmp = (std::max)(0.0,(double) i-ip);
              std::size_t indt = (std::size_t) tmp;
              for (std::size_t j=1;j<=mtheta_[i];j++) {
                std::size_t ij = igrid_[i] + j;
// upward
                double tdum = (j*deltat_[i]+zdum*(qtinv_[i]-qtinv_[indp]))/deltat_[indp];
                std::size_t jt = static_cast<std::size_t>(floor(tdum));
                double wt = tdum - jt;
                jt = (jt+mtheta_[indp])%(mtheta_[indp]);
                if ( ip == 1 ) {
                  wtp1_(1,ij,k) = wt;
                  jtp1_(1,ij,k) = static_cast<double>(igrid_[indp] + jt);
                } else {
                  wtp2_(1,ij,k) = wt;
                  jtp2_(1,ij,k) = static_cast<double>(igrid_[indp] + jt);
                }
// downward
                tdum = (j*deltat_[i]+zdum*(qtinv_[i]-qtinv_[indt]))/deltat_[indt];
                jt = static_cast<std::size_t>(floor(tdum));
                wt = tdum - jt;
                jt = (jt+mtheta_[indt])%mtheta_[indt];
                if ( ip == 1 ) {
                  wtp1_(2,ij,k) = wt;
                  jtp1_(2,ij,k) = static_cast<double>(igrid_[indt] + jt);
                } else {
                  wtp2_(2,ij,k) = wt;
                  jtp2_(2,ij,k) = static_cast<double>(igrid_[indt] + jt);
                }
              }
            }
          }
        }

    }

    void point::load(std::size_t objectid,parameter const& par)
    {
      // initialize random number generator
      srand((unsigned int)(objectid+5));

      double rmi = 1.0/(mi_*par->npartdom);
      double pi2_inv = 0.5/pi_;
      double w_initial = 1.0e-3;
      if ( par->nonlinear < 0.5 ) w_initial = 1.0e-12;
      std::size_t ntracer = 0;
      if ( objectid == 0 ) ntracer = 1;

      for (std::size_t m=1;m<=mi_;m++) {
        zion_(1,m,0) = sqrt(par->a0*par->a0 + ( (m+objectid*mi_)-0.5 )*(par->a1*par->a1-par->a0*par->a0)*rmi);
      } 

      if ( par->track_particles ) BOOST_ASSERT(false); // not implemented yet 

      // Set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
      // set random zion 
      for (std::size_t i=2;i<=6;i++) {
        for (std::size_t j=1;j<=mi_;j++) {
          zion_(i,j,0) = unifRand(0,1);
        }
      }

      // poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
      for (std::size_t m=1;m<=mi_;m++) {
        zion_(2,m,0) = 2.0*pi_*(zion_(2,m,0)-0.5);
        zion0_(2,m,0) = zion_(2,m,0); // zion0(2,:) for temporary storage
      }

      for (std::size_t i=1;i<=10;i++) {
        for (std::size_t m=1;m<=mi_;m++) {
          zion_(2,m,0) = zion_(2,m,0)*pi2_inv+10.0; // period of 1
          zion_(2,m,0) = 2.0*pi_*(zion_(2,m,0)-floor(zion_(2,m,0)));
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

      for (std::size_t m=1;m<=mi_;m++) {
        double z4tmp = zion_(4,m,0);
        zion_(4,m,0) = zion_(4,m,0)-0.5;
        if ( zion_(4,m,0) > 0.0 ) zion0_(4,m,0) = 1.0;
        else zion0_(4,m,0) = -1.0;
        zion_(4,m,0) = sqrt( (std::max)(SMALL,log(1.0/(std::max)(SMALL,pow(zion_(4,m,0),2)))));
        zion_(4,m,0) = zion_(4,m,0) - (c0+c1*zion_(4,m,0)+c2*pow(zion_(4,m,0),2))/
                                    (1.0+d1*zion_(4,m,0)+d2*pow(zion_(4,m,0),2)+d3*pow(zion_(4,m,0),3));
        if ( zion_(4,m,0) > par->umax ) zion_(4,m,0) = z4tmp;
      }

      for (std::size_t m=1;m<=mi_;m++) {
        // toroidal:  uniformly distributed in zeta
        zion_(3,m,0) = zetamin_+(zetamax_-zetamin_)*zion_(3,m,0);
        zion_(4,m,0) = zion0_(4,m,0)*(std::min)(par->umax,zion_(4,m,0));

        // initial random weight
        zion_(5,m,0) = 2.0*w_initial*(zion_(5,m,0)-0.5)*(1.0+cos(zion_(2,m,0)));

        // Maxwellian distribution in v_perp, <v_perp^2>=1.0
        zion_(6,m,0) = (std::max)(SMALL,(std::min)(par->umax*par->umax,-log((std::max)(SMALL,zion_(6,m,0)))));
      }

      // transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu)
      double vthi = par->gyroradius*abs(par->qion)/par->aion;
      for (std::size_t m=1;m<=mi_;m++) {
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
        for (std::size_t m=1;m<=mi_;m++) {
          zion0_(6,m,0) = 1.0;
        }
      } else {
        std::cerr << " Not implemented yet " << std::endl;
      }

      if ( par->nhybrid > 0 ) {
        std::cerr << " Not implemented yet " << std::endl;
      }
    }

    void point::chargei(std::size_t istep, std::vector<hpx::naming::id_type> const& point_components, parameter const& par)
    {

      double delr = 1.0/deltar_;
      double delz = 1.0/deltaz_;
      std::vector<double> delt;
      delt.resize( deltat_.size() );
      for (std::size_t i=0;i<deltat_.size();i++) {
        delt[i] = 2.0*pi_/deltat_[i];
      }
      double smu_inv = sqrt(par->aion)/(abs(par->qion)*par->gyroradius);
      double pi2_inv = 0.5/pi_;
      for (std::size_t i=0;i<=mzeta_;i++) {
        for (std::size_t j=0;j<=mgrid_;j++) {
          densityi_(i,j,0) = 0.0;
        }
      }
      std::size_t zero = 0;

      for (std::size_t m=1;m<=mi_;m++) {
        double psitmp = zion_(1,m,0);
        double thetatmp = zion_(2,m,0);
        double zetatmp = zion_(3,m,0);
        double rhoi = zion_(6,m,0)*smu_inv;

        double r = sqrt(2.0*psitmp);
        std::size_t tmp = (std::size_t) ((r-par->a0)*delr+0.5);
        double dip = (std::max)(0.0,(double) (std::min)(par->mpsi,tmp));
        std::size_t ip = (std::size_t) dip;
        std::size_t tmp2 = (std::size_t) (thetatmp*pi2_inv*delt[ip]+0.5);
        double djt = (std::max)(0.0,(double) (std::min)(mtheta_[ip],tmp2));
        std::size_t jt = (std::size_t) djt;
        std::size_t ipjt = igrid_[ip] + jt;

        double wz1 = (zetatmp-zetamin_)*delz;
        std::size_t kk = (std::max)(zero,(std::min)(par->mpsi-1,(std::size_t) wz1));
        kzion_[m] = static_cast<double>(kk);
        wzion_[m] = wz1 - kk;

        for (std::size_t larmor=1;larmor<=4;larmor++) {
          double rdum = delr*(std::max)(0.0,(std::min)(par->a1-par->a0,r+rhoi*pgyro_(larmor,ipjt,0)-par->a0));
          std::size_t ii = (std::max)(zero,(std::min)(par->mpsi-1,(std::size_t) rdum));
          double wp1 = rdum - ii;
          wpion_(larmor,m,0) = wp1;

          // particle position in theta
          double tflr = thetatmp + rhoi*tgyro_(larmor,ipjt,0);

          // inner flux surface
          std::size_t im = ii;
          double tdum = pi2_inv*(tflr-zetatmp*qtinv_[im])+10.0;
          tdum = (tdum - floor(tdum))*delt[im];
          std::size_t j00 = (std::max)(zero,(std::min)(mtheta_[im]-1,(std::size_t) tdum));
          jtion0_(larmor,m,0) = static_cast<double>(igrid_[im] + j00);
          wtion0_(larmor,m,0) = tdum - j00;

          // outer flux surface
          im = ii;
          tdum = pi2_inv*(tflr-zetatmp*qtinv_[im])+10.0;
          tdum = (tdum - floor(tdum))*delt[im];
          std::size_t j01 = (std::max)(zero,(std::min)(mtheta_[im]-1,(std::size_t) tdum));
          jtion1_(larmor,m,0) = static_cast<double>(igrid_[im] + j01);
          wtion1_(larmor,m,0) = tdum - j01;
        }
      }

      if ( istep == 0 ) return;

      for (std::size_t m=1;m<=mi_;m++) {
        double weight = zion_(5,m,0);

        std::size_t kk = static_cast<std::size_t>(kzion_[m]);
        double wz1 = weight*wzion_[m];
        double wz0 = weight-wz1;

        for (std::size_t larmor=1;larmor<=4;larmor++) {
          double wp1 = wpion_(larmor,m,0);
          double wp0 = 1.0-wp1;

          double wt10 = wp0*wtion0_(larmor,m,0);
          double wt00 = wp0 - wt10;

          double wt11 = wp1*wtion1_(larmor,m,0);
          double wt01 = wp1-wt11;

          std::size_t ij = static_cast<std::size_t>(jtion0_(larmor,m,0));
          densityi_(kk,ij,0) = densityi_(kk,ij,0) + wz0*wt00;
          densityi_(kk+1,ij,0) = densityi_(kk+1,ij,0) + wz1*wt00;

          ij = ij + 1;
          densityi_(kk,ij,0) = densityi_(kk,ij,0) + wz0*wt10;
          densityi_(kk+1,ij,0)   = densityi_(kk+1,ij,0)   + wz1*wt10;

          ij = static_cast<std::size_t>(jtion1_(larmor,m,0));
          densityi_(kk,ij,0) = densityi_(kk,ij,0) + wz0*wt01;
          densityi_(kk+1,ij,0)   = densityi_(kk+1,ij,0)   + wz1*wt01;

          ij = ij + 1;
          densityi_(kk,ij,0) = densityi_(kk,ij,0) + wz0*wt11;
          densityi_(kk+1,ij,0)   = densityi_(kk+1,ij,0)   + wz1*wt11;
        }
      }

      if ( par->npartdom > 1 ) {
        std::cerr << " Not implemented yet " << std::endl;
        BOOST_ASSERT(false);
      }

      // poloidal end cell, discard ghost cell j=0
      for (std::size_t i=0;i<=par->mpsi;i++) {
        for (std::size_t j=0;j<densityi_.isize();j++) {
          densityi_(j,igrid_[i]+mtheta_[i],0) = densityi_(j,igrid_[i]+mtheta_[i],0) + densityi_(j,igrid_[i],0);
        }
      }

      // send densityi to the left; receive from the right
      {
        typedef std::vector<hpx::lcos::promise< std::valarray<double> > > lazy_results_type;
        lazy_results_type lazy_results;
        lazy_results.push_back( stubs::point::get_densityi_async( point_components[right_pe_] ) );
        hpx::lcos::wait(lazy_results,
              boost::bind(&point::chargei_callback, this, _1, _2));
      }

      if ( myrank_toroidal_ == par->ntoroidal-1 ) {
        // B.C. at zeta=2*pi is shifted
        size_t strides[]= {1,1};
        std::size_t lengths[2];
        for (std::size_t i=0;i<=par->mpsi;i++) {
          std::size_t ii = igrid_[i];
          std::size_t jt = mtheta_[i];
          std::size_t start = ii+1;
          lengths[0]= jt+1;
          lengths[1]= 1; 
          std::gslice mygslice (start,std::valarray<size_t>(lengths,2),std::valarray<size_t>(strides,2)); 
          std::valarray<double> trecv = recvr_[mygslice];
          trecv.cshift(itran_[i]);
          for (std::size_t j=ii+1;j<=ii+jt;j++) {
            densityi_(mzeta_,j,0) += trecv[j-(ii+1)];
          }
        }
      } else {
        // B.C. at zeta<2*pi is continuous
        for (std::size_t j=0;j<densityi_.jsize();j++) {
          densityi_(mzeta_,j,0) += recvr_[j];
        }
      }

      // zero out charge in radial boundary cell
      for (std::size_t i=0;i<par->nbound;i++) {
        for (std::size_t j=0;j<densityi_.isize();j++) {
          for (std::size_t k=igrid_[i];k<igrid_[i]+mtheta_[i];k++) {
            densityi_(j,k,0) *= ((double) i)/par->nbound;
          } 
          for (std::size_t k=igrid_[par->mpsi-i];k<igrid_[par->mpsi-i]+mtheta_[par->mpsi-i];k++) {
            densityi_(j,k,0) *= ((double) i)/par->nbound;
          }
        }
      }

      // flux surface average and normalization
      std::fill( zonali_.begin(),zonali_.end(),0.0);

      for (std::size_t i=0;i<=par->mpsi;i++) {
        for (std::size_t j=1;j<=mtheta_[i];j++) {
          for (std::size_t k=1;k<=mzeta_;k++) {
            std::size_t ij = igrid_[i] + j; 
            zonali_[i] += 0.25*densityi_(k,ij,0);
            densityi_(k,ij,0) *= 0.25*markeri_(k,ij,0);
          }
        }
      }

      // global sum of phi00 broadcast to every toroidal PE
      {
        adum_.resize(zonali_.size());
        std::fill( adum_.begin(),adum_.end(),0.0);
        typedef std::vector<hpx::lcos::promise< std::vector<double> > > lazy_results_type;
        lazy_results_type lazy_results;
        BOOST_FOREACH(hpx::naming::id_type const& gid, point_components)
        {
          lazy_results.push_back( stubs::point::get_zonali_async( gid ) );
        }
        hpx::lcos::wait(lazy_results,
              boost::bind(&point::chargei_zonali_callback, this, _1, _2));
      }
      for (std::size_t i=0;i<zonali_.size();i++) {
        zonali_[i] = adum_[i]*pmarki_[i];
      }

      for (std::size_t i=0;i<=par->mpsi;i++) {
        for (std::size_t j=1;j<=mtheta_[i];j++) {
          for (std::size_t k=1;k<=mzeta_;k++) {
            std::size_t ij = igrid_[i] + j;
            densityi_(k,ij,0) -= zonali_[i];
          }
        }
        // poloidal BC condition
        for (std::size_t j=1;j<=mzeta_;j++) {
          densityi_(j,igrid_[i],0) = densityi_(j,igrid_[i]+mtheta_[i],0);
        } 
      } 

      // enforce charge conservation for zonal flow mode
      double rdum = 0.0;      
      double tdum = 0.0;
      for (std::size_t i=1;i<=par->mpsi-1;i++) {
        double r = par->a0 + deltar_*i;
        rdum += r;
        tdum += r*zonali_[i];
      }
      tdum /= rdum;
      for (std::size_t i=1;i<=par->mpsi-1;i++) {
        zonali_[i] -= tdum;
      }

    }

    bool point::chargei_zonali_callback(std::size_t i,std::vector<double> const& zonali)
    {
      for (std::size_t i=0;i<zonali.size();i++) {
        adum_[i] += zonali[i];
      }
      return true;
    }

    bool point::chargei_callback(std::size_t i,std::valarray<double> const& density)
    {
      recvr_.resize(density.size());
      recvr_ = density;
      return true;
    }

    std::valarray<double> point::get_densityi()
    {
      return densityi_.slicer(6,0);
    }

    std::vector<double> point::get_zonali()
    {
      return zonali_;
    }

    void point::smooth(std::size_t iflag, std::vector<hpx::naming::id_type> const& point_components, parameter const& par)
    {
      array<double> phitmp;
      phitmp.resize(mzeta_+1,mgrid_+1,1);
      for (std::size_t j=0;j<phitmp.jsize();j++) {
        for (std::size_t i=0;i<phitmp.isize();i++) {
          phitmp(i,j,0) = 0.0;
        }
      }  

      if ( iflag == 0 ) {
        for (std::size_t i=1;i<=mgrid_;i++) {
          for (std::size_t j=1;j<=mzeta_;j++) {
            phitmp(j,i,0) = densityi_(j,i,0);
          }
        }
      } else if ( iflag == 1 ) {
        for (std::size_t i=1;i<=mgrid_;i++) {
          for (std::size_t j=1;j<=mzeta_;j++) {
            phitmp(j,i,0) = densityi_(j,i,0);
          }
        }
      } else {
        for (std::size_t i=1;i<=mgrid_;i++) {
          for (std::size_t j=1;j<=mzeta_;j++) {
            phitmp(j,i,0) = phi_(j,i,0);
          }
        }
      }

      std::size_t ismooth;
      ismooth = 1;
      if ( par->nonlinear < 0.5 ) ismooth = 0;
      
      std::vector<double> phism;
      std::valarray<double> pright;
      phism.resize(mgrid_+1);
      pright.resize(par->mthetamax+1);

      for (std::size_t ip=1;ip<=ismooth;ip++) {
        // radial smoothing
        for (std::size_t i=1;i<=par->mpsi-1;i++) {
          for (std::size_t j=0;j<phitmp.isize();j++) {
            phitmp(j,igrid_[i],0) = phitmp(j,igrid_[i]+mtheta_[i],0);
          } 
        }

        for (std::size_t k=1;k<=mzeta_;k++) {
          std::fill( phism.begin(),phism.end(),0.0);
          for (std::size_t i=1;i<=par->mpsi-1;i++) {
            for (std::size_t j=1;j<=mtheta_[i];j++) {
              std::size_t ij = igrid_[i] + j;

              phism[ij] = 0.25*((1.0-wtp1_(1,ij,k))*phitmp(k,jtp1_(1,ij,k),0)+
                   wtp1_(1,ij,k)*phitmp(k,jtp1_(1,ij,k)+1,0)+
                   (1.0-wtp1_(2,ij,k))*phitmp(k,jtp1_(2,ij,k),0)+
                   wtp1_(2,ij,k)*phitmp(k,jtp1_(2,ij,k)+1,0))-
                   0.0625*((1.0-wtp2_(1,ij,k))*phitmp(k,jtp2_(1,ij,k),0)+
                   wtp2_(1,ij,k)*phitmp(k,jtp2_(1,ij,k)+1,0)+
                   (1.0-wtp2_(2,ij,k))*phitmp(k,jtp2_(2,ij,k),0)+
                   wtp2_(2,ij,k)*phitmp(k,jtp2_(2,ij,k)+1,0));
            }
          }

          for (std::size_t i=0;i<phism.size();i++) {
            phitmp(k,i,0) = 0.625*phitmp(k,i,0) + phism[i];
          }
        }

        // poloidal smoothing (-0.0625 0.25 0.625 0.25 -0.0625)
        for (std::size_t i=1;i<=par->mpsi-1;i++) {
          std::size_t ii = igrid_[i];
          std::size_t jt = mtheta_[i];
          for (std::size_t k=1;k<=mzeta_;k++) {
            for (std::size_t kk=1;kk<=jt;kk++) {
              pright[kk] = phitmp(k,ii+kk,0);
            }
            std::valarray<double> m_pright = pright.cshift(-1);
            std::valarray<double> p_pright = pright.cshift(1);
            std::valarray<double> m2_pright = pright.cshift(-2);
            std::valarray<double> p2_pright = pright.cshift(2);
            for (std::size_t kk=1;kk<=jt;kk++) {
              phitmp(k,ii+kk,0) = 0.625*pright[kk] +
                 0.25*(m_pright[kk]+p_pright[kk])-
                 0.0625*(m2_pright[kk]+p2_pright[kk]);
            }
          }
        }
 
        // parallel smoothing
      }

    }

}}

