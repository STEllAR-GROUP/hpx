//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_3735DA56_EFCE_4630_8FD7_DBA8CA194BBC)
#define HPX_3735DA56_EFCE_4630_8FD7_DBA8CA194BBC

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/particle.hpp"

namespace gtc
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a gtc::server::particle
    /// components.
    class particle
      : public hpx::components::client_base<particle, stubs::particle>
    {
        typedef hpx::components::client_base<particle, stubs::particle>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        particle()
        {}

        /// Create a client side representation for the existing
        /// \a gtc::server::particle instance with the given GID.
        particle(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::particle instance with the
        /// given particle file. 
        hpx::lcos::promise<void> init_async(std::size_t objectid,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,par);
        }

        /// Initialize the \a gtc::server::particle instance with the
        /// given particle file.  
        void init(std::size_t objectid,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init(gid_,objectid,par);
        }

        hpx::lcos::promise<void> chargei_async(std::size_t objectid,std::size_t istep, std::vector<hpx::naming::id_type> const& particle_components,
            parameter const& par)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::chargei_async(gid_,objectid,istep,particle_components,par);
        }

        void chargei(std::size_t objectid,std::size_t istep,std::vector<hpx::naming::id_type> const& particle_components,parameter const& par)
        {
            BOOST_ASSERT(gid_);
            this->base_type::chargei(gid_,objectid,istep,particle_components,par);
        }

        /// Compute the distance from the particle to the specified coordinates. 
        hpx::lcos::promise<double> distance_async(double posx, double posy,
            double posz)
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
        hpx::lcos::promise<std::size_t> get_index_async()
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

        /// Get the index of the particle.
        hpx::lcos::promise< array<double> > get_densityi_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_densityi_async(gid_);
        }

        /// Get the index of the particle.
        array<double> get_densityi()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_densityi(gid_);
        }
    };
}

#endif

