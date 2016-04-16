////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/version.hpp>
#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/function.hpp>

#include <boost/bind.hpp>

#include <string>

namespace hpx { namespace performance_counters
{
    void counter_type_shutdown(boost::shared_ptr<manage_counter_type> const& p)
    {
        error_code ec(lightweight);
        p->uninstall(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    counter_status install_counter_type(std::string const& name,
        counter_type type, std::string const& helptext,
        std::string const& uom, boost::uint32_t version, error_code& ec)
    {
        counter_info info(type, name, helptext,
            version ? version : HPX_PERFORMANCE_COUNTER_V1, uom);
        boost::shared_ptr<manage_counter_type> p =
            boost::make_shared<manage_counter_type>(info);

        // Install the counter type.
        p->install(ec);

        // Register the shutdown function which will clean up this counter type.
        get_runtime().add_shutdown_function(
            boost::bind(&counter_type_shutdown, p));
        return status_valid_data;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Install a new generic performance counter type in a way, which will
    // uninstall it automatically during shutdown.
    counter_status install_counter_type(std::string const& name,
        counter_type type, std::string const& helptext,
        create_counter_func const& create_counter,
        discover_counters_func const& discover_counters,
        boost::uint32_t version, std::string const& uom, error_code& ec)
    {
        counter_info info(type, name, helptext,
            version ? version : HPX_PERFORMANCE_COUNTER_V1, uom);
        boost::shared_ptr<manage_counter_type> p =
            boost::make_shared<manage_counter_type>(info);

        // Install the counter type.
        p->install(create_counter, discover_counters, ec);

        // Register the shutdown function which will clean up this counter type.
        get_runtime().add_shutdown_function(
            boost::bind(&counter_type_shutdown, p));
        return status_valid_data;
    }

    // Install a new generic performance counter type which uses a function to
    // provide the data in a way, which will uninstall it automatically during
    // shutdown.
    counter_status install_counter_type(std::string const& name,
        hpx::util::function_nonser<boost::int64_t(bool)> const& counter_value,
        std::string const& helptext, std::string const& uom, error_code& ec)
    {
        return install_counter_type(name, counter_raw, helptext,
            boost::bind(&hpx::performance_counters::locality_raw_counter_creator,
                _1, counter_value, _2),
            &hpx::performance_counters::locality_counter_discoverer,
            HPX_PERFORMANCE_COUNTER_V1, uom, ec);
    }

    /// Install several new performance counter types in a way, which will
    /// uninstall them automatically during shutdown.
    void install_counter_types(generic_counter_type_data const* data,
        std::size_t count, error_code& ec)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            install_counter_type(data[i].name_, data[i].type_,
                data[i].helptext_, data[i].create_counter_,
                data[i].discover_counters_, data[i].version_,
                data[i].unit_of_measure_, ec);
            if (ec) break;
        }
    }
}}

