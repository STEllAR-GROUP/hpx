//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assertion.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/register_thread.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/util/thread_description.hpp>

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
                "hpx::threads::get_self_or_default_pool",
                "Attempting to register a thread outside the HPX runtime and "
                "no default pool handler is installed. Did you mean to run "
                "this on an HPX thread?");
        }

        return pool;
    }
}}}    // namespace hpx::threads::detail
