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
            std::size_t scale,std::size_t number_partitions,double overlap)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_,objectid,scale,number_partitions,overlap);
        }

        // kernel 1
        void init(std::size_t objectid,std::size_t scale,std::size_t number_partitions,double overlap)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_,objectid, scale,number_partitions,overlap);
        }

        hpx::lcos::promise<void> root_async(std::vector<std::size_t> const& bfs_roots)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::root_async(gid_,bfs_roots);
        }

        void root(std::vector<std::size_t> const& bfs_roots)
        {
            BOOST_ASSERT(gid_);
            this->base_type::root(gid_,bfs_roots);
        }

        hpx::lcos::promise<void> bfs_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::bfs_async(gid_);
        }

        void bfs()
        {
            BOOST_ASSERT(gid_);
            this->base_type::bfs(gid_);
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

        hpx::lcos::promise< std::vector<nodedata> > validate_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::validate_async(gid_);
        }

        std::vector<nodedata> validate()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::validate(gid_);
        }

        hpx::lcos::promise< validatedata > scatter_async(std::vector<std::size_t> const&parent,
                                                         std::size_t searchkey,
                                                         std::size_t scale)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::scatter_async(gid_,parent,searchkey,scale);
        }

        validatedata scatter(std::vector<std::size_t> const&parent,
                             std::size_t searchkey,
                             std::size_t scale)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::scatter(gid_,parent,searchkey,scale);
        }

    };
}

#endif

