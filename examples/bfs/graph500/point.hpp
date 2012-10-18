//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4805)
#define HPX_68F602E3_C235_4660_AEAC_D5BD7AEC4805

#include <hpx/include/client.hpp>

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
            return this->base_type::init_async(get_gid(),objectid,scale,number_partitions,overlap);
        }

        // kernel 1
        void init(std::size_t objectid,std::size_t scale,std::size_t number_partitions,double overlap)
        {
            this->base_type::init_async(get_gid(),objectid, scale,number_partitions,overlap);
        }

        hpx::lcos::future<void> root_async(std::vector<int64_t> const& bfs_roots)
        {
            return this->base_type::root_async(get_gid(),bfs_roots);
        }

        void root(std::vector<int64_t> const& bfs_roots)
        {
            this->base_type::root(get_gid(),bfs_roots);
        }

        hpx::lcos::future<void> receive_duplicates_async(int64_t j,
                          std::vector<hpx::naming::id_type> const& duplicate_components,
                          std::vector<std::size_t> const& duplicateid)
        {
            return this->base_type::receive_duplicates_async(get_gid(),j,duplicate_components,duplicateid);
        }

        void receive_duplicates(int64_t j,
                          std::vector<hpx::naming::id_type> const& duplicate_components,
                          std::vector<std::size_t> const& duplicateid)
        {
            this->base_type::receive_duplicates(get_gid(),j,duplicate_components,duplicateid);
        }

        hpx::lcos::future<void> ppedge_async(int64_t start,int64_t stop,
                          std::vector<hpx::naming::id_type> const& point_components)
        {
            return this->base_type::ppedge_async(get_gid(),start,stop,point_components);
        }

        void ppedge(int64_t start,int64_t stop, 
                    std::vector<hpx::naming::id_type> const& point_components)
        {
            this->base_type::ppedge(get_gid(),start,stop,point_components);
        }

        hpx::lcos::future<void> bfs_async()
        {
            return this->base_type::bfs_async(get_gid());
        }

        void bfs()
        {
            this->base_type::bfs(get_gid());
        }

        hpx::lcos::future<void> resolve_conflict_async()
        {
            return this->base_type::resolve_conflict_async(get_gid());
        }

        void resolve_conflict()
        {
            this->base_type::resolve_conflict(get_gid());
        }

        hpx::lcos::future< std::vector<int> > distributed_validate_async(std::size_t scale)
        {
            return this->base_type::distributed_validate_async(get_gid(),scale);
        }

        std::vector<int> distributed_validate(std::size_t scale)
        {
            return this->base_type::distributed_validate(get_gid(),scale);
        }

        hpx::lcos::future< std::vector<bool> > findwhohasthisedge_async(int64_t edge,
                            std::vector<hpx::naming::id_type> const& point_components)
        {
            return this->base_type::findwhohasthisedge_async(get_gid(),edge,point_components);
        }

        std::vector<bool> findwhohasthisedge(int64_t edge,
                            std::vector<hpx::naming::id_type> const& point_components)
        {
            return this->base_type::findwhohasthisedge(get_gid(),edge,point_components);
        }

        hpx::lcos::future< std::vector<int64_t> > get_numedges_async()
        {
            return this->base_type::get_numedges_async(get_gid());
        }

        std::vector<int64_t> get_numedges()
        {
            return this->base_type::get_numedges(get_gid());
        }

        hpx::lcos::future< bool > has_edge_async(int64_t edge)
        {
            return this->base_type::has_edge_async(get_gid(),edge);
        }

        bool has_edge(int64_t edge)
        {
            return this->base_type::has_edge(get_gid(),edge);
        }

        hpx::lcos::future< resolvedata > get_parent_async(int64_t edge)
        {
            return this->base_type::get_parent_async(get_gid(),edge);
        }

        resolvedata get_parent(int64_t edge)
        {
            return this->base_type::get_parent(get_gid(),edge);
        }
    };
}

#endif

