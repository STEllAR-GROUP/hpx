//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/static.hpp>

#include <boost/foreach.hpp>

#if defined(__ANDROID__) && defined(ANDROID)
#include <cpu-features.h>
#endif

#if defined(_POSIX_VERSION)
#include <sys/syscall.h>
#include <sys/resource.h>
#endif

#if !defined(HPX_HAVE_HWLOC)

namespace hpx { namespace threads
{
    mask_type noop_topology::empty_mask = mask_type(hardware_concurrency());
}}

#else
#include <hwloc.h>
#include <hpx/exception.hpp>

namespace hpx { namespace threads { namespace detail
{
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
        return std::size_t(num_of_pus);
    }
}}}
#endif

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    mask_type topology::get_service_affinity_mask(
        mask_cref_type used_processing_units, error_code& ec) const
    {
        // We bind the service threads to the first NUMA domain. This is useful
        // as the first NUMA domain is likely to have the PCI controllers etc.
        mask_cref_type machine_mask = this->get_numa_node_affinity_mask(0, true, ec);
        if (ec || !any(machine_mask))
            return mask_type();

        if (&ec != &throws)
            ec = make_success_code();

        mask_type res = ~used_processing_units & machine_mask;

        return (!any(res)) ? machine_mask : res;
    }

    bool topology::reduce_thread_priority(error_code& ec) const
    {
#if defined(__linux__) && !defined(__ANDROID__)
        pid_t tid;
        tid = syscall(SYS_gettid);
        if (setpriority(PRIO_PROCESS, tid, 19))
        {
            HPX_THROWS_IF(ec, no_success, "threadmanager_impl::tfunc",
                "setpriority returned an error");
            return false;
        }
#elif defined(BOOST_MSVC)
        if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST))
        {
            HPX_THROWS_IF(ec, no_success, "threadmanager_impl::tfunc", 
                "SetThreadPriority returned an error");
            return false;
        }
#endif
        return true;
    }

    topology const& get_topology()
    {
        runtime* rt = get_runtime_ptr();
        if (rt == NULL)
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::threads::get_topology",
                "the hpx runtime system has not been initialized yet");
        }
        return rt->get_topology();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct hardware_concurrency_tag {};

    struct hw_concurrency
    {
        hw_concurrency()
#if defined(__ANDROID__) && defined(ANDROID)
          : num_of_cores_(::android_getCpuCount())
#elif defined(HPX_HAVE_HWLOC)
          : num_of_cores_(get_topology().hardware_concurrency())
#else
          : num_of_cores_(boost::thread::hardware_concurrency())
#endif
        {
            if (num_of_cores_ == 0)
                num_of_cores_ = 1;
        }

        std::size_t num_of_cores_;
    };

    std::size_t hardware_concurrency()
    {
        util::static_<hw_concurrency, hardware_concurrency_tag> hwc;
        return hwc.get().num_of_cores_;
    }
}}

