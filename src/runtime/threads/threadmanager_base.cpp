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
    // Return the number of the NUMA node the current thread is running on
    std::size_t get_numa_node_number()
    {
        std::size_t thread_num = hpx::get_worker_thread_num();
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
