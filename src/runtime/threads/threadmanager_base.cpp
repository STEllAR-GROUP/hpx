//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach, Katelyn Kufahl
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    hpx::util::thread_specific_ptr<std::size_t, threadmanager_base::tls_tag>
        threadmanager_base::thread_num_;

    void threadmanager_base::init_tss(std::size_t thread_num, bool numa_sensitive)
    {
        HPX_ASSERT(NULL == threadmanager_base::thread_num_.get());    // shouldn't be initialized yet
        threadmanager_base::thread_num_.reset(new std::size_t);
        if (numa_sensitive) {
            *threadmanager_base::thread_num_.get() =
                thread_num | (std::size_t(0x1) << 31);
        }
        else {
            *threadmanager_base::thread_num_.get() = thread_num;
        }
    }

    void threadmanager_base::deinit_tss()
    {
        threadmanager_base::thread_num_.reset();
    }

    std::size_t threadmanager_base::get_worker_thread_num(bool* numa_sensitive)
    {
        if (NULL != threadmanager_base::thread_num_.get())
        {
            std::size_t result = *threadmanager_base::thread_num_;
            if (std::size_t(-1) != result)
            {
                if (numa_sensitive)
                    *numa_sensitive = (result & (std::size_t(0x1) << 31)) != 0;
                return result & ~(std::size_t(0x1) << 31);
            }
        }

        // some OS threads are not managed by the thread-manager
        if (numa_sensitive)
            *numa_sensitive = false;
        return std::size_t(-1);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Return the number of the NUMA node the current thread is running on
    std::size_t get_numa_node_number()
    {
        bool numa_sensitive = false;
        std::size_t thread_num =
            threadmanager_base::get_worker_thread_num(&numa_sensitive);
        return get_topology().get_numa_node_number(
            get_thread_manager().get_pu_num(thread_num));
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::int64_t get_thread_count(thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state);
    }

    boost::int64_t get_thread_count(thread_priority priority,
        thread_state_enum state)
    {
        return get_thread_manager().get_thread_count(state, priority);
    }
}}
