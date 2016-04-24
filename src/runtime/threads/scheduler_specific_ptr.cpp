//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#include <hpx/runtime/threads/scheduler_specific_ptr.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/exception_ptr.hpp>

namespace hpx { namespace threads { namespace detail
{
    void* get_tss_data(void const* key)
    {
#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
        hpx::threads::thread_id_type self_id = hpx::threads::get_self_id();
        if (!self_id)
        {
            boost::throw_exception(coroutines::null_thread_id_exception());
            return 0;
        }
        return self_id->get_scheduler_base()->get_tss_data(key);
#endif
        return NULL;
    }

    void set_tss_data(void const* key,
        boost::shared_ptr<coroutines::detail::tss_cleanup_function> const& func,
        void* tss_data, bool cleanup_existing)
    {
#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
        hpx::threads::thread_id_type self_id = hpx::threads::get_self_id();
        if (!self_id)
        {
            boost::throw_exception(coroutines::null_thread_id_exception());
            return;
        }
        self_id->get_scheduler_base()->set_tss_data(key, func, tss_data,
            cleanup_existing);
#endif
    }
}}}
