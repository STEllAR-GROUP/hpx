//  Copyright (c) 2012 Vinay C Amatya
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/performance_counters.hpp>

#include <hpx/components/memory/mem_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE()

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace memory
{
    void register_counter_types()
    {
        namespace pc = hpx::performance_counters;
        pc::install_counter_type(
            "/memory/vm", &read_psm_vm,
            "returns the amount of virtual memory currently allocated by the "
            "referenced locality", "bytes"
        );
        pc::install_counter_type(
            "/memory/resident", &read_psm_resident,
            "returns the amount of resident memory currently allocated by the "
            "referenced locality", "bytes"
        );
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(HPX_STD_FUNCTION<void()>& startup_func)
    {
        // return our startup-function
        startup_func = register_counter_types;
        return true;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a px-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE(hpx::performance_counters::memory::get_startup);

