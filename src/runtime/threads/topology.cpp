////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime.hpp>

#if defined(__ANDROID__) && defined(ANDROID)
#include <cpu-features.h>
#endif

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    mask_type topology::get_service_affinity_mask(
        mask_type used_processing_units, error_code& ec) const
    {
        mask_type machine_mask = this->get_machine_affinity_mask(ec);
        if (ec || 0 == machine_mask)
            return 0;

        if (&ec != &throws)
            ec = make_success_code();

        return ~used_processing_units & machine_mask;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t hardware_concurrency()
    {
    #if defined(__ANDROID__) && defined(ANDROID)
        static std::size_t num_of_cores = ::android_getCpuCount();
    #else
        static std::size_t num_of_cores = boost::thread::hardware_concurrency();
    #endif

        if (0 == num_of_cores)
            return 1;           // Assume one core.

        return num_of_cores;
    }

    topology const& get_topology()
    {
        return get_runtime().get_topology();
    }
}}

