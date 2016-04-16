//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION_AUG_04_2011_0255PM)
#define HPX_PARTITION_AUG_04_2011_0255PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <string>

#include "../server/partition.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate1d { namespace stubs
{
    struct partition
      : hpx::components::stub_base<interpolate1d::server::partition>
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid, std::string datafilename,
            dimension const& dim, std::size_t num_nodes)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate1d::server::partition::init_action action_type;
            return hpx::async<action_type>(
                gid, datafilename, dim, num_nodes);
        }

        static void init(hpx::naming::id_type const& gid,
            std::string datafilename, dimension const& dim,
            std::size_t num_nodes)
        {
            init_async(gid, datafilename, dim, num_nodes).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<double>
        interpolate_async(hpx::naming::id_type const& gid, double value)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate1d::server::partition::interpolate_action action_type;
            return hpx::async<action_type>(gid, value);
        }

        static double interpolate(hpx::naming::id_type const& gid, double value)
        {
            return interpolate_async(gid, value).get();
        }
    };
}}

#endif


