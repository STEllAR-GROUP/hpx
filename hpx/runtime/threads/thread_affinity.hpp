//  Copyright (c) 2008-2009 Chirag Dekate, Hartmut Kaiser, Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREAD_AFFINITY_NOV_11_2008_0711PM)
#define HPX_RUNTIME_THREAD_AFFINITY_NOV_11_2008_0711PM

#include <hpx/hpx_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
#if defined(_WIN32) || defined(_WIN64)

    bool set_affinity(boost::thread& thrd, std::size_t num_thread)
    {
        unsigned int num_of_cores = boost::thread::hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        unsigned int affinity = num_thread % num_of_cores;

        DWORD_PTR process_affinity = 0, system_affinity = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process_affinity, 
              &system_affinity))
        {
            DWORD_PTR mask = 0x1 << affinity;
            while (!(mask & process_affinity)) {
                mask <<= 1;
                if (0 == mask)
                    mask = 0x01;
            }
            return SetThreadAffinityMask(thrd.native_handle(), mask) != 0;
        }
        return false;
    }
    inline bool set_affinity(std::size_t affinity)
    {
        return true;
    }

#elif defined(__APPLE__)

    #include <AvailabilityMacros.h>
    #include <mach/mach.h>
    #include <mach/mach_error.h>
#ifdef AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER
    #include <mach/thread_policy.h>
#endif

    inline bool set_affinity(boost::thread& thrd, std::size_t affinity)
    {
        return true;
    }

    inline bool set_affinity(std::size_t num_thread)
    {
#ifdef AVAILABLE_MAC_OS_X_VERSION_10_5_AND_LATER
        thread_affinity_policy_data_t policy;

        policy.affinity_tag = num_thread;
        kern_return_t ret = thread_policy_set(mach_thread_self(), 
            THREAD_AFFINITY_POLICY, (thread_policy_t) &policy, 
            THREAD_AFFINITY_POLICY_COUNT);

        return ret == KERN_SUCCESS;
#else
        return true;
#endif
    }

#else

    #include <sched.h>    // declares the scheduling interface

    inline bool set_affinity(boost::thread& thrd, std::size_t num_thread)
    {
        return true;
    }
    bool set_affinity(std::size_t num_thread)
    {
        std::size_t num_of_cores = boost::thread::hardware_concurrency();
        if (0 == num_of_cores)
            num_of_cores = 1;     // assume one core
        std::size_t affinity = num_thread % num_of_cores;

        cpu_set_t cpu;
        CPU_ZERO(&cpu);
        CPU_SET(affinity, &cpu);
        if (sched_setaffinity (0, sizeof(cpu), &cpu) == 0) {
            sleep(0);   // allow the OS to pick up the change
            return true;
        }
        return false;
    }
#endif

}}

#endif
