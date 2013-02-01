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
        // We bind the service threads to the first numa domain. This is useful
        // as the first numa domain is likely to have the PCI controllers etc.
        mask_type machine_mask = this->get_numa_node_affinity_mask(0, true, ec);
        if (ec || 0 == machine_mask)
            return 0;

        if (&ec != &throws)
            ec = make_success_code();

        mask_type res = ~used_processing_units & machine_mask;

        if(res == 0) return machine_mask;
        else return res;
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

