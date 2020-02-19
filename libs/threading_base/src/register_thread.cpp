//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assertion.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace hpx { namespace threads { namespace detail {
    static get_default_pool_type get_default_pool;

    void set_get_default_pool(get_default_pool_type f)
    {
        get_default_pool = f;
    }

    HPX_EXPORT thread_pool_base* get_self_or_default_pool()
    {
        thread_pool_base* pool = nullptr;
        auto thrd_data = get_self_id_data();
        if (thrd_data)
        {
            pool = thrd_data->get_scheduler_base()->get_parent_pool();
        }
        else if (detail::get_default_pool)
        {
            pool = detail::get_default_pool();
            HPX_ASSERT(pool);
        }
        else
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::threads::detail::get_self_or_default_pool",
                "Attempting to register a thread outside the HPX runtime and "
                "no default pool handler is installed. Did you mean to run "
                "this on an HPX thread?");
        }

        return pool;
    }

    static get_default_timer_service_type get_default_timer_service_f;

    void set_get_default_timer_service(get_default_timer_service_type f)
    {
        get_default_timer_service_f = f;
    }

    HPX_EXPORT boost::asio::io_service* get_default_timer_service()
    {
        boost::asio::io_service* timer_service = nullptr;
        if (detail::get_default_timer_service_f)
        {
            timer_service = detail::get_default_timer_service_f();
            HPX_ASSERT(timer_service);
        }
        else
        {
#if defined(HPX_HAVE_TIMER_POOL)
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::threads::detail::get_default_timer_service",
                "No timer service installed. When running timed threads "
                "without a runtime a timer service has to be installed "
                "manually using "
                "hpx::threads::detail::set_get_default_timer_service.");
#else
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::threads::detail::get_default_timer_service",
                "No timer service installed. Rebuild HPX with "
                "HPX_WITH_TIMER_POOL=ON or provide a timer service manually "
                "using hpx::threads::detail::set_get_default_timer_service.");
#endif
        }

        return timer_service;
    }
}}}    // namespace hpx::threads::detail
