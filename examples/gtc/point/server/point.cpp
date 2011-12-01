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

