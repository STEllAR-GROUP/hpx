//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_PARTICLE)
#define HPX_COMPONENTS_CLIENT_PARTICLE

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include "stubs/particle.hpp"

namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a stubs#particle class is the client side representation of all
    /// \a server#particle components
    class particle
        : public components::client_base<particle, stubs::particle>
    {
        typedef components::client_base<particle, stubs::particle>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component)
        particle()
        {}

        /// Create client side representation from a newly create component
        /// instance.
        particle(naming::id_type where, double x, double y)
          : base_type(base_type::create_sync(where))    // create component
        {
            //init(x, y);   // initialize coordinates
        }

        /// Create a client side representation for the existing
        /// \a server#particle instance with the given global id \a gid.
        particle(naming::id_type gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the server#particle instance with the given \a gid
        lcos::promise<void> init_async(std::size_t objectid,std::string graphfile)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,graphfile);
        }

        void init(std::size_t objectid,std::string graphfile)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,objectid,graphfile);
        }

        lcos::promise<double> distance_async(double posx, double posy, double posz)
        {
          BOOST_ASSERT(gid_);
          return this->base_type::distance_async(gid_,posx,posy,posz);
        }

        double distance(double posx, double posy, double posz) 
        {
          BOOST_ASSERT(gid_);
          return this->base_type::distance(gid_,posx,posy,posz);
        }

    };
}}

#endif
