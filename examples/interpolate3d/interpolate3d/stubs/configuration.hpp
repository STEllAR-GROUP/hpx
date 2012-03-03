//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE3D_CONFIGURATION_AUG_07_2011_0648PM)
#define HPX_INTERPOLATE3D_CONFIGURATION_AUG_07_2011_0648PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/configuration.hpp"

namespace interpolate3d { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct configuration
      : hpx::components::stubs::stub_base<interpolate3d::server::configuration>
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<void>
        init_async(hpx::naming::id_type const& gid, std::string const& datafile,
            std::string const& symbolic_name, std::size_t num_instances)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate3d::server::configuration::init_action action_type;
            return hpx::lcos::eager_future<action_type>(
                gid, datafile, symbolic_name, num_instances);
        }

        static void init(hpx::naming::id_type const& gid,
            std::string const& datafile, std::string const& symbolic_name,
            std::size_t num_instances)
        {
            init_async(gid, datafile, symbolic_name, num_instances).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::promise<config_data>
        get_async(hpx::naming::id_type const& gid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized promise, the caller needs
            // to call get() on the return value to obtain the result
            typedef interpolate3d::server::configuration::get_action action_type;
            return hpx::lcos::eager_future<action_type>(gid);
        }

        static config_data get(hpx::naming::id_type const& gid)
        {
            return get_async(gid).get();
        }
    };
}}

#endif
