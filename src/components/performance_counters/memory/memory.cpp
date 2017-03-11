//  Copyright (c) 2012 Vinay C Amatya
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_startup_shutdown.hpp>
#include <hpx/util/function.hpp>

#include <hpx/components/performance_counters/memory/mem_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality, We register the module dynamically
// as no executable links against it.
HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace memory
{
    void register_counter_types()
    {
        namespace pc = hpx::performance_counters;
        pc::install_counter_type(
            "/runtime/memory/virtual", &read_psm_virtual,
            "returns the amount of virtual memory currently allocated by the "
            "referenced locality", "bytes"
        );
        pc::install_counter_type(
            "/runtime/memory/resident", &read_psm_resident,
            "returns the amount of resident memory currently allocated by the "
            "referenced locality", "bytes"
        );
        pc::install_counter_type(
            "/runtime/memory/total", &read_total_mem_avail,
            "returns the total available memory on the node", "kB"
        );
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(hpx::startup_function_type& startup_func, bool& pre_startup)
    {
        // return our startup-function
        startup_func = register_counter_types;    // function to run during startup
        pre_startup = true;  // run 'register_counter_types' as pre-startup function
        return true;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE_DYNAMIC(
    hpx::performance_counters::memory::get_startup);

