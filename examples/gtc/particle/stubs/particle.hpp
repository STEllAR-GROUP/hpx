//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_PARTICLE)
#define HPX_COMPONENTS_STUBS_PARTICLE

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/particle.hpp"

namespace hpx { namespace geometry { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct particle : components::stub_base<server::particle>
    {
        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a hpx::geometry::server::particle instance with the
        /// given particle file.  
        static lcos::promise<void>
        init_async(naming::id_type const& gid,std::size_t objectid,
            std::string const& particlefile)
        {
            typedef server::particle::init_action action_type;
            return lcos::eager_future<action_type>(gid,objectid,particlefile);
        }

        /// Initialize the \a hpx::geometry::server::particle instance with the
        /// given particle file.  
        static void init(naming::id_type const& gid,std::size_t objectid,
            std::string const& particlefile)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            init_async(gid,objectid,particlefile).get();
        }

        /// Compute the distance from the particle to the specified coordinates. 
        static lcos::promise<double>
        distance_async(naming::id_type const& gid,double posx,double posy,double posz) 
        {
            typedef server::particle::distance_action action_type;
            return lcos::eager_future<action_type>(gid,posx,posy,posz);
        } 

        /// Compute the distance from the particle to the specified coordinates. 
        static double distance(naming::id_type const& gid,double posx,double posy,double posz)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return distance_async(gid,posx,posy,posz).get();
        }

        /// Get the index of the particle.
        static lcos::promise<std::size_t> get_index_async(naming::id_type const& gid)
        {
            typedef server::particle::get_index_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        /// Get the index of the particle.
        static std::size_t get_index(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return get_index_async(gid).get();
        }
    };
}}}

#endif

