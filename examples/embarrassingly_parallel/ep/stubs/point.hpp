//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_85F90266_2241_4547_A2A5_2D1E88242D14)
#define HPX_85F90266_2241_4547_A2A5_2D1E88242D14

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

namespace ep { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct point : hpx::components::stub_base<server::point>
    {
        static hpx::lcos::promise<void>
        bfs_async(hpx::naming::id_type const& gid,std::size_t scale)
        {
            typedef server::point::bfs_action action_type;
            return hpx::lcos::eager_future<action_type>(gid,scale);
        }

        static void bfs(hpx::naming::id_type const& gid,std::size_t scale)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise.
            bfs_async(gid,scale).get();
        }
    };
}}

#endif

