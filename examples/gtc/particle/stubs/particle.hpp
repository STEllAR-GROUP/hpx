//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_PARTICLE)
#define HPX_COMPONENTS_STUBS_PARTICLE

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/particle.hpp"

namespace hpx { namespace geometry { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#particle class is the client side representation of all
    /// \a server#particle components
    struct particle : components::stub_base<server::particle>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#particle instance with the given \a gid
        static lcos::promise<void>
        init_async(naming::id_type gid,std::size_t objectid,std::string graphfile)
        {
            typedef server::particle::init_action action_type;
            return lcos::eager_future<action_type>(gid,objectid,graphfile);
        }

        static void init(naming::id_type const& gid,std::size_t objectid,std::string graphfile)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            init_async(gid,objectid,graphfile).get();
        }

        static lcos::promise<double> distance_async(naming::id_type const& gid,double posx,double posy, double posz) 
        {
          typedef server::particle::distance_action action_type;
          return lcos::eager_future<action_type>(gid,posx,posy,posz);
        } 

        static double distance(naming::id_type const& gid, double posx,double posy, double posz)
        {
          return distance_async(gid,posx,posy,posz).get();
        }
    };

}}}

#endif
