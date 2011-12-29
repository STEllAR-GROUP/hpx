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

#include "../../particle/stubs/particle.hpp"
#include "point.hpp"

#include <string>
#include <sstream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::init(std::size_t objectid,parameter const& par)
    {
        idx_ = objectid;

        tauii_ = -1.0; // initially collisionless

        // initial mesh
        std::size_t toroidal_domain_location=objectid/par->npartdom;
        double pi = 4.0*atan(1.0);
        mzeta_ = par->mzetamax/par->ntoroidal;
        double tmp1 = (double) toroidal_domain_location;
        double tmp2 = (double) par->ntoroidal;
        double tmp3 = (double) (toroidal_domain_location+1);
        zetamin_ = 2.0*pi*tmp1/tmp2;
        zetamax_ = 2.0*pi*tmp3/tmp2;

        double tmp4 = (double) mzeta_;
        deltaz_ = (zetamax_-zetamin_)/tmp4;

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
        double tdum = 2.0*pi*par->a1/tmp6;

        // initial data
        for (std::size_t i=0;i<par->mpsi+1;i++) {
          double r = par->a0 + deltar_*i; 
          std::size_t two = 2;
          double tmp7 = pi*r/tdum + 0.5;
          std::size_t tmp8 = (std::size_t) tmp7;
          mtheta_[i] = std::max(two,std::min(par->mthetamax,two*tmp8)); // even # poloidal grid
          deltat_[i] = 2.0*pi/mtheta_[i];
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
        std::size_t mi = par->micell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of ions per processor
        if ( mi <  (mi_local%par->npartdom) ) mi++;
        std::size_t me_local = par->mecell*(mgrid_-par->mpsi)*mzeta_;  // # of electrons in toroidal domain
        std::size_t me = par->mecell*(mgrid_-par->mpsi)*mzeta_/par->npartdom; // # of electrons per processor
        if ( me < (me_local%par->npartdom) ) me++;

        double tmp13 = (double) mi;
        double tmp14 = (double) me;
        std::size_t mimax = mi + 100*std::ceil(sqrt(tmp13)); // ions array upper bound
        std::size_t memax = me + 100*std::ceil(sqrt(tmp14)); // electrons array upper bound

        pgyro_.resize(5,mgrid_+1,1);
        tgyro_.resize(5,mgrid_+1,1);
        markeri_.resize(mzeta_+1,mgrid_+1,1);
        densityi_.resize(mzeta_+1,mgrid_,1);
        phi_.resize(mzeta_+1,mgrid_,1);
        evector_.resize(3,mzeta_+1,mgrid_);
        jtp1_.resize(2,mgrid_,mzeta_);
        jtp2_.resize(2,mgrid_,mzeta_);
        wtp1_.resize(2,mgrid_,mzeta_);
        wtp2_.resize(2,mgrid_,mzeta_);
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
          double rmax = std::min(par->a1,r+0.5*deltar_); 
          double rmin = std::max(par->a0,r-0.5*deltar_); 
          double tmp15 = (double) mi*par->npartdom;
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
        zion_.resize(nparam,mimax,1);
        zion0_.resize(nparam,mimax,1);
        jtion0_.resize(nparam,mimax,1);
        jtion1_.resize(4,mimax,1);
        kzion_.resize(mimax);
        wzion_.resize(mimax);
        wpion_.resize(4,mimax,1);
        wtion0_.resize(4,mimax,1);
        wtion1_.resize(4,mimax,1);

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
    }

    bool search_callback(std::list<std::size_t>& deposits,
        std::size_t i,double const& distance)
    {
        double neighbor_distance = 0.1;
        if ( distance < neighbor_distance ) {
            // deposit the charge of this particle on the gridpoint
            deposits.push_back(i); 
        }
        return true; 
    }

    void point::search(std::vector<hpx::naming::id_type> const& particle_components)
    {
        // For demonstration, a simple search strategy: we check if the
        // particle is within a certain distance of the gridpoint.  If so, then
        // get its charge
        typedef std::vector<hpx::lcos::promise<double> > lazy_results_type;

        lazy_results_type lazy_results;

        BOOST_FOREACH(hpx::naming::id_type const& gid, particle_components)
        {
            lazy_results.push_back( stubs::particle::distance_async( gid,posx_,posy_,posz_ ) );
        }

        // List of particles whose charge should deposited on this gridpoint. 
        std::list<std::size_t> deposits;

        // Wait on the results, and invoke a callback when each result is ready.
        hpx::lcos::wait(lazy_results,
            boost::bind(&search_callback, boost::ref(deposits), _1, _2));

        // Print out the particles whose charge should be deposited on this
        // point.
        BOOST_FOREACH(std::size_t i, deposits)
        {
            hpx::cout << ( boost::format("deposit particle %1% on point %2%\n")
                         % idx_ % stubs::particle::get_index(particle_components.at(i)))
                      << hpx::flush; 
        }
    }
}}

