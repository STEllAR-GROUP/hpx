//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_POINT)
#define HPX_COMPONENTS_STUBS_POINT

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

namespace gtc { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct point : hpx::components::stub_base<server::point>
    {
        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::point instance with the
        /// given point file. 
        static hpx::lcos::promise<void>
        init_async(hpx::naming::id_type gid,std::size_t objectid,
            std::size_t max_num_neighbors,std::string const& meshfile)
        {
            typedef server::point::init_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,objectid,
                max_num_neighbors,meshfile);
        }

        /// Initialize the \a gtc::server::point instance with the
        /// given point file.  
        static void init(hpx::naming::id_type const& gid,std::size_t objectid,
            std::size_t max_num_neighbors,std::string const& meshfile)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            init_async(gid,objectid,max_num_neighbors,meshfile).get();
        }

        /// Perform a search on the \a gtc::server::point
        /// components in \a particle_components. 
        static hpx::lcos::promise<void>
        search_async(hpx::naming::id_type gid,
            std::vector<hpx::naming::id_type> const& particle_components)
        {
            typedef server::point::search_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,particle_components);
        }

        /// Perform a search on the \a gtc::server::point
        /// components specified \a particle_components. 
        static void search(hpx::naming::id_type const& gid,
            std::vector<hpx::naming::id_type> const& particle_components)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            search_async(gid,particle_components).get();
        }
    };
}}

#endif

