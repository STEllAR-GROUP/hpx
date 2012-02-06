//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_85F90266_2241_4547_A2A5_2D1E88242D14)
#define HPX_85F90266_2241_4547_A2A5_2D1E88242D14

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

namespace graph500 { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct point : hpx::components::stub_base<server::point>
    {
        // Read the graph
        static hpx::lcos::promise<void>
        init_async(hpx::naming::id_type const& gid,std::size_t objectid,std::size_t scale,
                   std::size_t number_partitions)
        {
            typedef server::point::init_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,objectid,scale,
                                             number_partitions);
        }

        // Read the graph
        static void init(hpx::naming::id_type const& gid,std::size_t objectid,std::size_t scale,
                         std::size_t number_partitions)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            init_async(gid,objectid,scale,number_partitions).get();
        }

        static hpx::lcos::promise<void>
        bfs_async(hpx::naming::id_type const& gid,std::size_t root)
        {
            typedef server::point::bfs_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,root);
        }

        static void bfs(hpx::naming::id_type const& gid,std::size_t root)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            bfs_async(gid,root).get();
        }

        static hpx::lcos::promise<void>
        reset_async(hpx::naming::id_type const& gid)
        {
            typedef server::point::reset_action action_type;
            return hpx::lcos::eager_future<action_type>(gid);
        }

        static void reset(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            reset_async(gid).get();
        }

        static hpx::lcos::promise< bool >
        has_edge_async(hpx::naming::id_type const& gid,std::size_t edge)
        {
            typedef server::point::has_edge_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,edge);
        }

        static bool has_edge(hpx::naming::id_type const& gid,std::size_t edge)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return has_edge_async(gid,edge).get();
        }

        static hpx::lcos::promise< std::vector<nodedata> >
        validate_async(hpx::naming::id_type const& gid)
        {
            typedef server::point::validate_action action_type;
            return hpx::lcos::eager_future<action_type>(gid);
        }

        static std::vector<nodedata> validate(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return validate_async(gid).get();
        }

        static hpx::lcos::promise< validatedata >
        scatter_async(hpx::naming::id_type const& gid,std::vector<std::size_t> const&parent,
                      std::size_t searchkey,std::size_t scale)
        {
            typedef server::point::scatter_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,parent,searchkey,scale);
        }

        static validatedata scatter(hpx::naming::id_type const& gid,
                                    std::vector<std::size_t> const&parent,
                                    std::size_t searchkey,std::size_t scale)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            return scatter_async(gid,parent,searchkey,scale).get();
        }

    };
}}

#endif

