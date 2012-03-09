//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_PARTICLE)
#define HPX_COMPONENTS_STUBS_PARTICLE

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/particle.hpp"

namespace gtc { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct particle : hpx::components::stub_base<server::particle>
    {
        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::particle instance with the
        /// given particle file.  
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid,std::size_t objectid,
            hpx::components::gtc::parameter const& par)
        {
            typedef server::particle::init_action action_type;
            return hpx::lcos::async<action_type>(gid,objectid,
                par);
        }

        /// Initialize the \a gtc::server::particle instance with the
        /// given particle file.  
        static void init(hpx::naming::id_type const& gid,std::size_t objectid,
            hpx::components::gtc::parameter const& par)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            init_async(gid,objectid,par).get();
        }

        static hpx::lcos::future<void>
        chargei_async(hpx::naming::id_type const& gid,std::size_t objectid,std::size_t istep,std::vector<hpx::naming::id_type> const& particle_components,
            hpx::components::gtc::parameter const& par)
        {
            typedef server::particle::chargei_action action_type;
            return hpx::lcos::async<action_type>(gid,objectid,istep,
                particle_components,par);
        }

        static void chargei(hpx::naming::id_type const& gid,std::size_t objectid,std::size_t istep,std::vector<hpx::naming::id_type> const& particle_components,
            hpx::components::gtc::parameter const& par)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            chargei_async(gid,objectid,istep,particle_components,par).get();
        }

        /// Compute the distance from the particle to the specified coordinates. 
        static hpx::lcos::future<double>
        distance_async(hpx::naming::id_type const& gid,double posx,double posy,
            double posz) 
        {
            typedef server::particle::distance_action action_type;
            return hpx::lcos::async<action_type>(gid,posx,posy,posz);
        } 

        /// Compute the distance from the particle to the specified coordinates. 
        static double distance(hpx::naming::id_type const& gid,double posx,
            double posy,double posz)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            return distance_async(gid,posx,posy,posz).get();
        }

        /// Get the index of the particle.
        static hpx::lcos::future<std::size_t>
        get_index_async(hpx::naming::id_type const& gid)
        {
            typedef server::particle::get_index_action action_type;
            return hpx::lcos::async<action_type>(gid);
        }

        /// Get the index of the particle.
        static std::size_t get_index(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return get_index_async(gid).get();
        }

        static hpx::lcos::future< array<double> >
        get_densityi_async(hpx::naming::id_type const& gid)
        {
            typedef server::particle::get_densityi_action action_type;
            return hpx::lcos::async<action_type>(gid);
        }

        static array<double> get_densityi(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return get_densityi_async(gid).get();
        }
    };
}}

#endif

