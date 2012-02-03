//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4805)
#define HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4805

#include <hpx/runtime/components/client_base.hpp>

#include "stubs/point.hpp"

namespace graph500
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a graph500::server::point components.
    class point : public hpx::components::client_base<point, stubs::point>
    {
        typedef hpx::components::client_base<point, stubs::point>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        point()
        {}

        /// Create a client side representation for the existing
        /// \a graph500::server::point instance with the given GID.
        point(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        // kernel 1
        hpx::lcos::promise<void> init_async(std::size_t objectid,
            std::size_t scale,std::size_t number_partitions)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,scale,number_partitions);
        }

        // kernel 1
        void init(std::size_t objectid,std::size_t scale,std::size_t number_partitions)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid, scale,number_partitions);
        }

        hpx::lcos::promise<void> bfs_async(std::size_t root)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::bfs_async(gid_,root);
        }

        void bfs(std::size_t root)
        {
            BOOST_ASSERT(gid_);
            this->base_type::bfs(gid_,root);
        }

        hpx::lcos::promise<void> reset_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::reset_async(gid_);
        }

        void reset()
        {
            BOOST_ASSERT(gid_);
            this->base_type::reset(gid_);
        }

        hpx::lcos::promise< bool > has_edge_async(std::size_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::has_edge_async(gid_,edge);
        }

        bool has_edge(std::size_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::has_edge(gid_,edge);
        }

    };
}

#endif

