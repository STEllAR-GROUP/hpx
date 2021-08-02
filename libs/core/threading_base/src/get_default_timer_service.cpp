//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/errors/error.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/threading_base/detail/get_default_timer_service.hpp>

#include <asio/io_context.hpp>

namespace hpx { namespace threads { namespace detail {
    static get_default_timer_service_type get_default_timer_service_f;

    void set_get_default_timer_service(get_default_timer_service_type f)
    {
        get_default_timer_service_f = f;
    }

    asio::io_context* get_default_timer_service()
    {
        asio::io_context* timer_service = nullptr;
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
                "No timer service installed. When running timed "
                "threads without a runtime a timer service has to be "
                "installed manually using "
                "hpx::threads::detail::set_get_default_timer_service.");
#else
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::threads::detail::get_default_timer_service",
                "No timer service installed. Rebuild HPX with "
                "HPX_WITH_TIMER_POOL=ON or provide a timer service "
                "manually using "
                "hpx::threads::detail::set_get_default_timer_service.");
#endif
        }

        return timer_service;
    }
}}}    // namespace hpx::threads::detail
