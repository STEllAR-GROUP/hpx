//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_85F90267_2241_4547_A2A5_2D1E88242D14)
#define HPX_85F90267_2241_4547_A2A5_2D1E88242D14

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

namespace ad { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct point : hpx::components::stub_base<server::point>
    {
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid,std::size_t scale,std::size_t np)
        {
            typedef server::point::init_action action_type;
            return hpx::async<action_type>(gid,scale,np);
        }

        static void init(hpx::naming::id_type const& gid,std::size_t scale,std::size_t np)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            init_async(gid,scale,np).get();
        }

        static hpx::lcos::future<void>
        remove_item_async(hpx::naming::id_type const& gid,std::size_t scale,std::size_t np)
        {
            typedef server::point::remove_item_action action_type;
            return hpx::async<action_type>(gid,scale,np);
        }

        static void remove_item(hpx::naming::id_type const& gid,std::size_t scale,std::size_t np)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            remove_item_async(gid,scale,np).get();
        }

        static hpx::lcos::future<void>
        calcrhs_async(hpx::naming::id_type const& gid)
        {
            typedef server::point::calcrhs_action action_type;
            return hpx::async<action_type>(gid);
        }

        static void calcrhs(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            calcrhs_async(gid).get();
        }

        static hpx::lcos::future<void>
        compute_async(hpx::naming::id_type const& gid,
                std::vector<hpx::naming::id_type> const& point_components)
        {
            typedef server::point::compute_action action_type;
            return hpx::async<action_type>(gid,point_components);
        }

        static void compute(hpx::naming::id_type const& gid,
                     std::vector<hpx::naming::id_type> const& point_components)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            compute_async(gid,point_components).get();
        }

        static hpx::lcos::future<std::size_t>
        get_item_async(hpx::naming::id_type const& gid)
        {
            typedef server::point::get_item_action action_type;
            return hpx::async<action_type>(gid);
        }

        static std::size_t get_item(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            return get_item_async(gid).get();
        }
    };
}}

#endif

