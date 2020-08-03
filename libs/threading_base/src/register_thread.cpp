//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

// The following implementation has been divided for Linux and Mac OSX
#if defined(HPX_HAVE_DYNAMIC_HPX_MAIN) &&                                      \
    (defined(__linux) || defined(__linux__) || defined(linux) ||               \
        defined(__APPLE__))

namespace hpx_start {
    // Redefining weak variables defined in hpx_main.hpp to facilitate error
    // checking and make sure correct errors are thrown. It is added again
    // to make sure that these variables are defined correctly in cases
    // where hpx_main functionalities are not used.
    HPX_SYMBOL_EXPORT bool is_linked __attribute__((weak)) = false;
    HPX_SYMBOL_EXPORT bool include_libhpx_wrap __attribute__((weak)) = false;
}    // namespace hpx_start

#endif

namespace hpx { namespace threads { namespace detail {
    static get_default_pool_type get_default_pool;

    void set_get_default_pool(get_default_pool_type f)
    {
        get_default_pool = f;
    }

    thread_pool_base* get_self_or_default_pool()
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
// The following implementation has been divided for Linux and Mac OSX
#if defined(HPX_HAVE_DYNAMIC_HPX_MAIN) &&                                      \
    (defined(__linux) || defined(__linux__) || defined(linux) ||               \
        defined(__APPLE__))

                    // hpx_main.hpp is included but not linked to libhpx_wrap
                    if (!hpx_start::is_linked && hpx_start::include_libhpx_wrap)
                        HPX_THROW_EXCEPTION(invalid_status,
                            "hpx::threads::detail::get_self_or_default_pool",
                            "Attempting to use hpx_main.hpp functionality "
                            "without "
                            "linking to libhpx_wrap. If you're using "
                            "CMakeLists, make "
                            "sure to add HPX::wrap_main to "
                            "target_link_libraries. "
                            "If you're using Makefile, make sure to link to "
                            "libhpx_wrap when generating the executable. If "
                            "you're "
                            "linking explicitly, consult the HPX docs for "
                            "library "
                            "link order and other subtle nuances.");

#endif

                    HPX_THROW_EXCEPTION(invalid_status,
                        "hpx::threads::detail::get_self_or_default_pool",
                        "Attempting to register a thread outside the HPX "
                        "runtime and "
                        "no default pool handler is installed. Did you mean to "
                        "run "
                        "this on an HPX thread?");
                }

                return pool;
            }

            static get_default_timer_service_type get_default_timer_service_f;

            void set_get_default_timer_service(get_default_timer_service_type f)
            {
                get_default_timer_service_f = f;
            }

            boost::asio::io_service* get_default_timer_service()
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
                        "No timer service installed. When running timed "
                        "threads "
                        "without a runtime a timer service has to be installed "
                        "manually using "
                        "hpx::threads::detail::set_get_default_timer_service.");
#else
                    HPX_THROW_EXCEPTION(invalid_status,
                        "hpx::threads::detail::get_default_timer_service",
                        "No timer service installed. Rebuild HPX with "
                        "HPX_WITH_TIMER_POOL=ON or provide a timer service "
                        "manually "
                        "using "
                        "hpx::threads::detail::set_get_default_timer_service.");
#endif
                }

                return timer_service;
            }
}}}    // namespace hpx::threads::detail
