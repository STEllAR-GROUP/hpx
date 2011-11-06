////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach, Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>
#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/actions/continuation.hpp>

namespace hpx { namespace performance_counters
{
    void counter_type_shutdown(boost::shared_ptr<manage_counter_type> const& p)
    {
        error_code ec;
        p->uninstall(ec);
    }

    void install_counter_type(std::string const& name,
        counter_type type, std::string const& helptext, boost::uint32_t version,
        error_code& ec)
    {
        counter_info info(type, name, helptext, version ? version : 0x01000000);
        boost::shared_ptr<manage_counter_type> p =
            boost::make_shared<manage_counter_type>(info);

        // Install the counter type.
        p->install(ec);

        // Register the shutdown function which will clean up this counter type.
        get_runtime().add_shutdown_function(
            boost::bind(&counter_type_shutdown, p));
    }

    void install_counter_types(raw_counter_type_data const* data,
        std::size_t count, error_code& ec)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            install_counter_type(data[i].name_, data[i].type_,
                data[i].helptext_, data[i].version_, ec);
            if (ec) break;
        }
    }
}}

