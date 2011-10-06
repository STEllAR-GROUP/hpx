//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREAD_AFFINITY_NOV_11_2008_0711PM)
#define HPX_RUNTIME_THREAD_AFFINITY_NOV_11_2008_0711PM

#include <hpx/hpx_fwd.hpp>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    inline std::size_t least_significant_bit(std::size_t mask)
    {
        if (mask) {
            int c = 0;    // will count v's trailing zero bits

            // Set mask's trailing 0s to 1s and zero rest
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c)
                mask >>= 1;

            return std::size_t(1) << c;
        }
        return std::size_t(1);
    }

    inline std::size_t least_significant_bit_set(std::size_t mask)
    {
        if (mask) {
            std::size_t c = 0;    // will count v's trailing zero bits

            // Set mask's trailing 0s to 1s and zero rest
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c)
                mask >>= 1;

            return c;
        }
        return std::size_t(-1);
    }

    inline std::size_t hardware_concurrency()
    {
        static std::size_t num_of_cores = boost::thread::hardware_concurrency();
        return num_of_cores;
    }

#if defined(BOOST_WINDOWS)

    inline int get_numa_node(std::size_t thread_num, bool numa_sensitive)
    {
        if (std::size_t(-1) == thread_num) 
             return -1;

        UCHAR node_number = 0;
        if (GetNumaProcessorNode(thread_num, &node_number))
            return int(node_number);

        unsigned int num_of_cores = hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core

        unsigned int num_of_numa_cores = num_of_cores;
        ULONG numa_nodes = 0;
        if (GetNumaHighestNodeNumber(&numa_nodes) && 0 != numa_nodes) 
            num_of_numa_cores = num_of_cores / (numa_nodes + 1);

        return thread_num / num_of_numa_cores;
    }

    inline std::size_t 
    get_numa_node_affinity_mask(std::size_t num_thread, bool numa_sensitive)
    {
        unsigned int num_of_cores = hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        std::size_t affinity = num_thread % num_of_cores;

        ULONG numa_nodes = 1;
        if (GetNumaHighestNodeNumber(&numa_nodes))
            ++numa_nodes;

        DWORD_PTR node_affinity_mask = 0;
        if (numa_sensitive) {
            ULONG numa_node = affinity % numa_nodes;
            if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
                return false;

            return node_affinity_mask;
        }

        ULONG numa_node = get_numa_node(num_thread, numa_sensitive);
        if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
            return false;

        return node_affinity_mask;
    }

    inline std::size_t 
    get_thread_affinity_mask(std::size_t num_thread, bool numa_sensitive)
    {
        unsigned int num_of_cores = hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        std::size_t affinity = num_thread % num_of_cores;

        ULONG numa_nodes = 1;
        if (GetNumaHighestNodeNumber(&numa_nodes))
            ++numa_nodes;

        std::size_t num_of_cores_per_numa_node = num_of_cores / numa_nodes;
        DWORD_PTR node_affinity_mask = 0;
        DWORD_PTR mask = 0x01LL;
        if (numa_sensitive) {
            ULONG numa_node = affinity % numa_nodes;
            if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
                return false;

            mask = least_significant_bit(node_affinity_mask) << (affinity / numa_nodes);
        }
        else {
            ULONG numa_node = get_numa_node(num_thread, numa_sensitive);
            if (!GetNumaNodeProcessorMask(numa_node, &node_affinity_mask))
                return false;

            mask = least_significant_bit(node_affinity_mask) << (affinity % num_of_cores_per_numa_node);
        }

        while (!(mask & node_affinity_mask)) {
            mask <<= 1LL;
            if (0 == mask)
                mask = 0x01LL;
        }

        return mask;
    }

    inline bool 
    set_affinity(boost::thread& thrd, std::size_t num_thread, bool numa_sensitive)
    {
        DWORD_PTR mask = get_thread_affinity_mask(num_thread, numa_sensitive);
        return SetThreadAffinityMask(thrd.native_handle(), mask) != 0;
    }

    inline bool set_affinity(std::size_t affinity, bool numa_sensitive)
    {
        return true;
    }

#elif defined(__APPLE__)

    // the thread affinity code is taken from the example:
    // http://www.opensource.apple.com/darwinsource/projects/other/xnu-1228.3.13/tools/tests/affinity/pool.c

    #include <AvailabilityMacros.h>
    #include <mach/mach.h>
    #include <mach/mach_error.h>
#ifdef AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER
    #include <mach/thread_policy.h>
#endif

    inline bool set_affinity(boost::thread& thrd, std::size_t affinity, 
        bool numa_sensitive)
    {
        return true;
    }
    inline bool set_affinity(std::size_t num_thread, bool numa_sensitive)
    {
#ifdef AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER
        thread_extended_policy_data_t epolicy;
        epolicy.timeshare = FALSE;

        kern_return_t ret = thread_policy_set(mach_thread_self(), 
            THREAD_EXTENDED_POLICY, (thread_policy_t) &epolicy,
            THREAD_EXTENDED_POLICY_COUNT);

        if (ret != KERN_SUCCESS)
            return false;

        thread_affinity_policy_data_t policy;
        policy.affinity_tag = num_thread + 1;   // 1...N

        ret = thread_policy_set(mach_thread_self(), 
            THREAD_AFFINITY_POLICY, (thread_policy_t) &policy, 
            THREAD_AFFINITY_POLICY_COUNT);

        return ret == KERN_SUCCESS;
#else
        return true;
#endif
    }

    inline std::size_t 
    get_thread_affinity_mask(std::size_t thread_num, bool numa_sensitive)
    {
        return std::size_t(-1);
    }

    inline std::size_t 
    get_numa_node_affinity_mask(std::size_t thread_num, bool numa_sensitive)
    {
        return std::size_t(-1);
    }

    inline int get_numa_node(std::size_t thread_num, bool numa_sensitive)
    {
        return -1;
    }

#else

    #include <pthread.h>
    #include <sched.h>    // declares the scheduling interface
    #include <sys/syscall.h>
    #include <sys/types.h>

    inline bool set_affinity(boost::thread& thrd, std::size_t num_thread, 
        bool numa_sensitive)
    {
        return true;
    }
    inline bool set_affinity(std::size_t num_thread, bool numa_sensitive)
    {
        std::size_t num_of_cores = hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core

        cpu_set_t cpu;
        CPU_ZERO(&cpu);

        // limit this thread to one of the cores
        std::size_t num_blocks = 1;
        std::size_t num_cores = num_of_cores;

        num_blocks = boost::lexical_cast<std::size_t>(
            get_config_entry("system_topology.num_blocks", num_blocks));
        num_cores = boost::lexical_cast<std::size_t>(
            get_config_entry("system_topology.num_cores", num_cores));

        // Check sanity
        assert(num_blocks * num_cores == num_of_cores);

        // Choose thread mapping function and determine affinity
        std::string thread_map = "linear";
        thread_map = get_config_entry("system_topology.thread_map", thread_map);

        std::size_t affinity(0);
        if (thread_map == "linear")
        {
          affinity = (num_thread) % num_of_cores;
        }
        else if (thread_map == "block_striped")
        {
          affinity = (num_thread/num_blocks)
                      + (num_cores * (num_thread % num_blocks));
        }
        else
        {
          assert(0);
        }

        CPU_SET(affinity, &cpu);
#if defined(HPX_HAVE_PTHREAD_AFFINITY_NP)
        if (0 == pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu))
#else
        if (0 == sched_setaffinity(syscall(SYS_gettid), sizeof(cpu), &cpu))
#endif
        {
            sleep(0);   // allow the OS to pick up the change
            return true;
        }
        return false;
    }

    inline std::size_t 
    get_thread_affinity_mask(std::size_t thread_num, bool numa_sensitive)
    {
        return std::size_t(-1);
    }

    inline std::size_t 
    get_numa_node_affinity_mask(std::size_t thread_num, bool numa_sensitive)
    {
        return std::size_t(-1);
    }

    inline int get_numa_node(std::size_t thread_num, bool numa_sensitive)
    {
        return -1;
    }

#endif

}}

#endif

