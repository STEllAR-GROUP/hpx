//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CLIENT_POINT)
#define HPX_COMPONENTS_CLIENT_POINT

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace hpx { namespace geometry
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a hpx::geometry::server::point
    /// components.
    class point : public components::client_base<point, stubs::point>
    {
        typedef components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        point()
        {}

        /// Create a client side representation for the existing
        /// \a hpx::geometry::server::point instance with the given GID.
        point(naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a hpx::geometry::server::point instance with the
        /// given point file. 
        lcos::promise<void> init_async(std::size_t objectid,
            std::size_t max_num_neighbors,std::string const& meshfile)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,max_num_neighbors,
                meshfile);
        }

        /// Initialize the \a hpx::geometry::server::point instance with the
        /// given point file.  
        void init(std::size_t objectid,std::size_t max_num_neighbors,
            std::string const& meshfile)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid,max_num_neighbors,
                meshfile);
        }

        /// Perform a search on the \a hpx::geometry::server::particle
        /// components specified. 
        lcos::promise<int> search_async(std::vector<hpx::naming::id_type> const& particle_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::search_async(gid_,particle_components);
        }

        /// Perform a search on the \a hpx::geometry::server::particle
        /// components specified. 
        int search(std::vector<hpx::naming::id_type> const& particle_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::search(gid_,particle_components);
        }

    };
}}

#endif

