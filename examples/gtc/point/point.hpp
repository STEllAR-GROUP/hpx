//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA)
#define HPX_AA0F78C3_B11C_4173_8FA8_C1A6073FB9BA

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace gtc
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a gtc::server::point
    /// components.
    class point
      : public hpx::components::client_base<point, stubs::point>
    {
        typedef hpx::components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        point()
        {}

        /// Create a client side representation for the existing
        /// \a gtc::server::point instance with the given GID.
        point(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a gtc::server::point instance with the
        /// given point file. 
        hpx::lcos::promise<void> init_async(std::size_t objectid,
            std::size_t max_num_neighbors,std::string const& meshfile)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,max_num_neighbors,
                meshfile);
        }

        /// Initialize the \a gtc::server::point instance with the
        /// given point file.  
        void init(std::size_t objectid,std::size_t max_num_neighbors,
            std::string const& meshfile)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid,max_num_neighbors,
                meshfile);
        }

        /// Perform a search on the \a gtc::server::particle
        /// components specified. 
        hpx::lcos::promise<void>
        search_async(std::vector<hpx::naming::id_type> const& particle_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::search_async(gid_,particle_components);
        }

        /// Perform a search on the \a gtc::server::particle
        /// components specified. 
        void search(std::vector<hpx::naming::id_type> const& particle_components)
        {
            BOOST_ASSERT(gid_);
            this->base_type::search(gid_,particle_components);
        }

    };
}

#endif

