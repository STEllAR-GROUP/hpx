//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2009 Oliver Kowalke
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/runtime/threads/coroutines/detail/context_base.hpp>

#if defined(HPX_HAVE_APEX)
#include <hpx/runtime/threads/thread_id_type.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#endif

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    // initialize static allocation counter
    allocation_counters context_base::m_allocation_counters;

#if defined(HPX_HAVE_APEX)
    // adding this here, because the thread_id_type and thread_data types
    // aren't fully defined in the header.
    void * rebind_base_apex(void * apex_data_ptr, thread_id_type id) {
        return ::hpx::util::apex_update_task(apex_data_ptr, 
            id.get()->get_description());
    }
#endif
}}}}
