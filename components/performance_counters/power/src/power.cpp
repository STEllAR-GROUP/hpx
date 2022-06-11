//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/components_base/component_startup_shutdown.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime_configuration/component_factory_base.hpp>
#include <hpx/runtime_local/startup_function.hpp>

#include <hpx/components/performance_counters/power/power_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality, We register the module dynamically
// as no executable links against it.
HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()

///////////////////////////////////////////////////////////////////////////////
namespace hpx::performance_counters::power {

    void register_counter_types()
    {
        hpx::performance_counters::install_counter_type(
            "/runtime/average_power", &average_power_consumption,
            "returns the average power consumption of the specified locality",
            "W", hpx::performance_counters::counter_type::raw);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(
        hpx::startup_function_type& startup_func, bool& pre_startup)
    {
        // return our startup-function

        // function to run during startup
        startup_func = register_counter_types;

        // run 'register_counter_types' as pre-startup function
        pre_startup = true;
        return true;
    }
}    // namespace hpx::performance_counters::power

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE_DYNAMIC(
    hpx::performance_counters::power::get_startup)
