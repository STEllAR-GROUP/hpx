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
        hpx::lcos::future<void> init_async(std::size_t objectid,
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

        hpx::lcos::future<void> root_async(std::vector<int64_t> const& bfs_roots)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::root_async(gid_,bfs_roots);
        }

        void root(std::vector<int64_t> const& bfs_roots)
        {
            BOOST_ASSERT(gid_);
            this->base_type::root(gid_,bfs_roots);
        }

        hpx::lcos::future<void> receive_duplicates_async(int64_t j,
                          std::vector<hpx::naming::id_type> const& duplicate_components,
                          std::vector<std::size_t> const& duplicateid)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::receive_duplicates_async(gid_,j,duplicate_components,duplicateid);
        }

        void receive_duplicates(int64_t j,
                          std::vector<hpx::naming::id_type> const& duplicate_components,
                          std::vector<std::size_t> const& duplicateid)
        {
            BOOST_ASSERT(gid_);
            this->base_type::receive_duplicates(gid_,j,duplicate_components,duplicateid);
        }

        hpx::lcos::future<void> ppedge_async(int64_t start,int64_t stop,
                          std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::ppedge_async(gid_,start,stop,point_components);
        }

        void ppedge(int64_t start,int64_t stop, 
                    std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            this->base_type::ppedge(gid_,start,stop,point_components);
        }

        hpx::lcos::future<void> bfs_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::bfs_async(gid_);
        }

        void bfs()
        {
            BOOST_ASSERT(gid_);
            this->base_type::bfs(gid_);
        }

        hpx::lcos::future<void> resolve_conflict_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::resolve_conflict_async(gid_);
        }

        void resolve_conflict()
        {
            BOOST_ASSERT(gid_);
            this->base_type::resolve_conflict(gid_);
        }

        hpx::lcos::future< std::vector<int> > distributed_validate_async(std::size_t scale)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::distributed_validate_async(gid_,scale);
        }

        std::vector<int> distributed_validate(std::size_t scale)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::distributed_validate(gid_,scale);
        }

        hpx::lcos::future< std::vector<bool> > findwhohasthisedge_async(int64_t edge,
                            std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::findwhohasthisedge_async(gid_,edge,point_components);
        }

        std::vector<bool> findwhohasthisedge(int64_t edge,
                            std::vector<hpx::naming::id_type> const& point_components)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::findwhohasthisedge(gid_,edge,point_components);
        }

        hpx::lcos::future< std::vector<int64_t> > get_numedges_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_numedges_async(gid_);
        }

        std::vector<int64_t> get_numedges()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_numedges(gid_);
        }

        hpx::lcos::future< bool > has_edge_async(int64_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::has_edge_async(gid_,edge);
        }

        bool has_edge(int64_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::has_edge(gid_,edge);
        }

        hpx::lcos::future< resolvedata > get_parent_async(int64_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_parent_async(gid_,edge);
        }

        resolvedata get_parent(int64_t edge)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_parent(gid_,edge);
        }
    };
}

#endif

