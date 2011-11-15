//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_PARTICLE)
#define HPX_COMPONENTS_CLIENT_PARTICLE

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/particle.hpp"

namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a hpx::geometry::server::particle
    /// components.
    class particle : public components::client_base<particle, stubs::particle>
    {
        typedef components::client_base<particle, stubs::particle>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        particle()
        {}

        /// Create a client side representation for the existing
        /// \a hpx::geometry::server::particle instance with the given GID.
        particle(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a hpx::geometry::server::particle instance with the
        /// given particle file. 
        lcos::promise<void> init_async(std::size_t objectid,std::string const& particlefile)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,particlefile);
        }

        /// Initialize the \a hpx::geometry::server::particle instance with the
        /// given particle file.  
        void init(std::size_t objectid,std::string const& particlefile)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,objectid,particlefile);
        }

        /// Compute the distance from the particle to the specified coordinates. 
        lcos::promise<double> distance_async(double posx, double posy, double posz)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::distance_async(gid_,posx,posy,posz);
        }

        /// Compute the distance from the particle to the specified coordinates. 
        double distance(double posx, double posy, double posz) 
        {
            BOOST_ASSERT(gid_);
            return this->base_type::distance(gid_,posx,posy,posz);
        }

        /// Get the index of the particle.
        lcos::promise<std::size_t> get_index_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_index_async(gid_);
        }

        /// Get the index of the particle.
        std::size_t get_index()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_index(gid_);
        }
    };
}}

#endif

