//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/lcos.hpp>

#include "../stubs/particle.hpp"

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

// namespace hpx { namespace traits
// {
//     template <typename T>
//     struct promise_remote_result<array<T> >
//     {
//         typedef array<T> type;
//     };
// }}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void particle::init(std::size_t objectid,parameter const& par)
    {
        idx_ = objectid;
        //
        srand((unsigned int)(objectid+5));

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

        deltat_.resize(par->mpsi+1);

        // grid shift associated with fieldline following coordinates
        double tmp6 = (double) par->mthetamax;
        double tdum = 2.0*pi*par->a1/tmp6;
        qtinv_.resize(par->mpsi+1);
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i;
          std::size_t two = 2;
          double tmp7 = pi*r/tdum + 0.5;
          std::size_t tmp8 = (std::size_t) tmp7;
          mtheta_[i] = (std::max)(two,(std::min)(par->mthetamax,two*tmp8)); // even # poloidal grid
          deltat_[i] = 2.0*pi/mtheta_[i];
          double q = par->q0 + par->q1*r/par->a + par->q2*r*r/(par->a*par->a);
          double tmp9 = mtheta_[i]/q + 0.5;
          double tmp11 = (double) mtheta_[i];
          qtinv_[i] = tmp11/tmp9;  // q value for coordinate transformation
          qtinv_[i] = 1.0/qtinv_[i]; // inverse q to avoid divide operation
        }

        // number of grids on a poloidal plane
        mgrid_ = 0;
        for (std::size_t i=0;i<mtheta_.size();i++) {
          mgrid_ += mtheta_[i] + 1;
        }
        mzeta_ = par->mzetamax/par->ntoroidal;
        std::size_t mi_ = par->micell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of ions per processor

        double rmi = 1.0/(mi_*par->npartdom);
        double pi2_inv = 0.5/pi;
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
        double tmp13 = (double) mi_;
        std::size_t mimax = mi_ + static_cast<std::size_t>(100*std::ceil(sqrt(tmp13))); // ions array upper bound

        zion_.resize(nparam+1,mimax+1,1);
        zion0_.resize(nparam+1,mimax+1,1);
        jtion0_.resize(nparam+1,mimax+1,1);
        jtion1_.resize(5,mimax+1,1);
        wtion1_.resize(5,mimax+1,1);
        for (std::size_t m=1;m<=mi_;m++) {
          zion_(1,m,0) = sqrt(par->a0*par->a0 + ( (m+objectid*mi_)-0.5 )*(par->a1*par->a1-par->a0*par->a0)*rmi);
        }

        if ( par->track_particles ) BOOST_ASSERT(false); // not implemented yet

        // Set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
        for (std::size_t i=2;i<=6;i++) {
          for (std::size_t j=1;j<=mi_;j++) {
            zion_(i,j,0) = unifRand(0,1);
          }
        }

        // poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
        for (std::size_t m=1;m<=mi_;m++) {
          zion_(2,m,0) = 2.0*pi*(zion_(2,m,0)-0.5);
          zion0_(2,m,0) = zion_(2,m,0); // zion0(2,:) for temporary storage
        }

        for (std::size_t i=1;i<=10;i++) {
          for (std::size_t m=1;m<=mi_;m++) {
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

        // memory/setup related to ion gather-scatter coefficients
        igrid_.resize(par->mpsi+1);
        igrid_[0] = 1;
        for (std::size_t i=1;i<par->mpsi+1;i++) {
          igrid_[i] = igrid_[i-1]+mtheta_[i-1]+1;
        }

        double tmp4 = (double) mzeta_;
        deltaz_ = (zetamax_-zetamin_)/tmp4;
        kzion_.resize(mimax+1);
        wzion_.resize(mimax+1);
        wtion0_.resize(5,mimax+1,1);
        wpion_.resize(5,mimax+1,1);

        pgyro_.resize(5,mgrid_+1,1);
        tgyro_.resize(5,mgrid_+1,1);
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
        densityi_.resize(mzeta_+1,mgrid_+1,1);
    }

    bool particle::chargei_callback(std::size_t i,array<double> const& density)
    {
      if ( i != idx_ ) {
        for (std::size_t ij=1;ij<=mgrid_;ij++) {
          for (std::size_t kk=0;kk<=mzeta_;kk++) {
            densityi_(kk,ij,0) += density(kk,ij,0);
          }
        }
      }
      return true;
    }

    void particle::chargei(std::size_t objectid,std::size_t istep,std::vector<hpx::naming::id_type> const& particle_components,parameter const& par)
    {
        // calculate ion gather-scatter coefficients
        double pi = 4.0*atan(1.0);

        double delr = 1.0/deltar_;
        double delz = 1.0/deltaz_;
        std::vector<double> delt;
        delt.resize( deltat_.size() );
        for (std::size_t i=0;i<deltat_.size();i++) {
          delt[i] = 2.0*pi/deltat_[i];
        }
        double smu_inv = sqrt(par->aion)/(abs(par->qion)*par->gyroradius);
        double pi2_inv = 0.5/pi;
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
          kzion_[m] = kk;
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

          std::size_t kk = kzion_[m];
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
          // All reduce on densityi
          typedef std::vector<hpx::lcos::future< array<double> > > lazy_results_type;

          lazy_results_type lazy_results;
          BOOST_FOREACH(hpx::naming::id_type const& gid, particle_components)
          {
            lazy_results.push_back( stubs::particle::get_densityi_async( gid ) );
          }
          hpx::lcos::wait(lazy_results,
                boost::bind(&particle::chargei_callback, this, _1, _2));
        }

        // poloidal end cell, discard ghost cell j=0
        for (std::size_t i=0;i<=par->mpsi;i++) {
          for (std::size_t j=0;j<densityi_.isize();j++) {
          densityi_(j,igrid_[i]+mtheta_[i],0) = densityi_(j,igrid_[i]+mtheta_[i],0) + densityi_(j,igrid_[i],0);
          }
        }

        // toroidal end cell
        // send idensity information to the left; receive from right
        // toroidal mesh

    }

    array<double> particle::get_densityi()
    {
      return densityi_;
    }

}}


