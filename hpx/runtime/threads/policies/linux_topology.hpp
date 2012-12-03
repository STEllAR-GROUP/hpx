////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c)      2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_66CC9C17_C5C8_412E_8C2C_0C991C83A5C1)
#define HPX_66CC9C17_C5C8_412E_8C2C_0C991C83A5C1

#if defined(HPX_HAVE_PTHREAD_SETAFFINITY_NP)
    #include <pthread.h>
#endif

#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/exception.hpp>

namespace hpx { namespace threads
{

struct linux_topology : topology
{
    std::size_t get_numa_node_number(
        std::size_t thread_num
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return std::size_t(-1);
    }

    std::size_t get_numa_node_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }

    std::size_t get_thread_affinity_mask(
        std::size_t thread_num
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }

    void set_thread_affinity(
        boost::thread& thrd
      , std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();
    }

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#endif
    void set_thread_affinity(
        std::size_t num_thread
      , bool numa_sensitive
      , error_code& ec = throws
        ) const
    {
        cpu_set_t cpu;

        CPU_ZERO(&cpu);

        CPU_SET(num_thread % hardware_concurrency(), &cpu);

        #if defined(HPX_HAVE_PTHREAD_SETAFFINITY_NP)
            if (0 == pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu))
                sleep(0); // Allow the OS to pick up the change.
        #else
            if (0 == sched_setaffinity(syscall(SYS_gettid), sizeof(cpu), &cpu))
                sleep(0); // Allow the OS to pick up the change.
        #endif

        else
        {
            HPX_THROWS_IF(ec, kernel_error
              , "hpx::threads::set_thread_affinity"
              , "failed to set thread affinity");
        }

        if (ec)
            return;
        else if (&ec != &throws)
            ec = make_success_code();
    }
#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic pop
#endif

    std::size_t get_thread_affinity_mask_from_lva(
        naming::address::address_type lva
      , error_code& ec = throws
        ) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        return 0;
    }
};

///////////////////////////////////////////////////////////////////////////////
inline topology* create_topology()
{
    return new linux_topology;
}

}}

#endif // HPX_66CC9C17_C5C8_412E_8C2C_0C991C83A5C1

