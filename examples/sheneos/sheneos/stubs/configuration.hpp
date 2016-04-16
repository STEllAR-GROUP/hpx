//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1217PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1217PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <string>

#include "../server/configuration.hpp"

namespace sheneos { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct configuration
      : hpx::components::stub_base<sheneos::server::configuration>
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<void>
        init_async(hpx::naming::id_type const& gid, std::string const& datafile,
            std::string const& symbolic_name, std::size_t num_instances)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            typedef sheneos::server::configuration::init_action action_type;
            return hpx::async<action_type>(
                gid, datafile, symbolic_name, num_instances);
        }

        static void init(hpx::naming::id_type const& gid,
            std::string const& datafile, std::string const& symbolic_name,
            std::size_t num_instances)
        {
            init_async(gid, datafile, symbolic_name, num_instances).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static hpx::lcos::future<config_data>
        get_async(hpx::naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future.
            typedef sheneos::server::configuration::get_action action_type;
            return hpx::async<action_type>(gid);
        }

        static config_data get(hpx::naming::id_type const& gid)
        {
            return get_async(gid).get();
        }
    };
}}

#endif
