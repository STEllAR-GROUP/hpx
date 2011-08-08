//  Copyright (c) 2011 Vinay C Amatya
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40)
#define HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include <nqueen/server/board.hpp>

namespace hpx { namespace nqueen { namespace stubs
{

    struct board : components::stub_base<server::board>
    {
        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<void> print_async(naming::id_type const& gid)
        {
            typedef server::board::print_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static void print(naming::id_type const& gid)
        {
            print_async(gid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<std::vector<std::size_t> >
        access_async(naming::id_type const& gid)
        {
            typedef server::board::access_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static std::vector<std::size_t> access(naming::id_type const& gid)
        {
            return access_async(gid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<void> update_async(naming::id_type const& gid,
            std::size_t level, std::size_t pos)
        {
            typedef server::board::update_action action_type;
            return lcos::eager_future<action_type>(gid, level, pos);
        }

        static void update(naming::id_type const& gid, std::size_t level,
            std::size_t pos)
        {
            update_async(gid, level, pos).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<bool> check_async(naming::id_type const& gid,
            std::vector<std::size_t> const& list, std::size_t level)
        {
            typedef server::board::check_action action_type;
            return lcos::eager_future<action_type>(gid, list, level);
        }

        static bool check(naming::id_type const& gid,
            std::vector<std::size_t> const& list, std::size_t level)
        {
            return check_async(gid, list, level).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<void> solve_async(naming::id_type const& gid,
            std::vector<std::size_t> const& list, std::size_t size,
            std::size_t level)
        {
            typedef server::board::solve_action action_type;
            return lcos::eager_future<action_type>(gid, list, size, level);
        }

        static void solve(naming::id_type const& gid,
            std::vector<std::size_t> const& list, std::size_t size,
            std::size_t level)
        {
            solve_async(gid, list, size, level).get(); 
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<void> clear_async(naming::id_type const& gid)
        {
            typedef server::board::clear_action action_type;
            return lcos::eager_future<action_type>(gid);
        }

        static void clear(naming::id_type const& gid)
        {
            clear_async(gid).get(); 
        }
    };
}}}

#endif // HPX_E4B0BA36_0E1C_48F5_928B_CDC78F1D2C40

