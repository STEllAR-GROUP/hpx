////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2012-2013 Thomas Heller
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
    
#if defined(HPX_HAVE_HWLOC)
#include <hwloc.h>
#include <hpx/exception.hpp>

namespace {
    std::size_t hwloc_hardware_concurrency()
    {
        hwloc_topology_t topo;
        int err = hwloc_topology_init(&topo);
        if (err != 0)
        {
            HPX_THROW_EXCEPTION(hpx::no_success, "hwloc_hardware_concurrency",
                "Failed to init hwloc hwloc_topology");
        }

        err = hwloc_topology_load(topo);
        if (err != 0)
        {
            hwloc_topology_destroy(topo);
            HPX_THROW_EXCEPTION(hpx::no_success, "hwloc_hardware_concurrency",
                "Failed to load hwloc topology");
        }
        int num_of_pus = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_PU);
        if(num_of_pus < 0)
        {
            hwloc_topology_destroy(topo);
            HPX_THROW_EXCEPTION(hpx::no_success, "hwloc_hardware_concurrency",
                "Failed to get number of PUs");
        }

        hwloc_topology_destroy(topo);

        return num_of_pus;
    }
}
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

    topology const& get_topology()
    {
        return get_runtime().get_topology();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t hardware_concurrency()
    {
    #if defined(__ANDROID__) && defined(ANDROID)
        static std::size_t num_of_cores = ::android_getCpuCount();
        
    #else
    #  if defined(HPX_HAVE_HWLOC)
        static std::size_t
            num_of_cores = ::hwloc_hardware_concurrency();
    #  else
        static std::size_t
            num_of_cores = boost::thread::hardware_concurrency();
    #  endif
    #endif

        if (0 == num_of_cores)
            return 1;           // Assume one core.

        return num_of_cores;
    }
}}

