//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2006-7 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/lcos/local/recursive_mutex.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

namespace hpx { namespace lcos { namespace local
{
    bool recursive_mutex::try_lock()
    {
        threads::thread_id_type const current_thread_id = threads::get_self_id();

        return try_recursive_lock(current_thread_id) ||
            try_basic_lock(current_thread_id);
    }

    void recursive_mutex::lock()
    {
        threads::thread_id_type const current_thread_id = threads::get_self_id();

        if (!try_recursive_lock(current_thread_id))
        {
            mtx.lock();
            locking_thread_id.exchange(current_thread_id);
            recursion_count = 1;
        }
    }

    bool recursive_mutex::timed_lock(::boost::system_time const& wait_until)
    {
        threads::thread_id_type const current_thread_id = threads::get_self_id();

        return try_recursive_lock(current_thread_id) ||
            try_timed_lock(current_thread_id, wait_until);
    }
}}}

