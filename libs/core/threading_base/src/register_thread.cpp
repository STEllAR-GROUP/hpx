//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2018 Thomas Heller
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_init_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

namespace hpx::threads {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        threads::thread_result_type cleanup_thread()
        {
            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks held.
            util::force_error_on_lock();

            // run and free all registered exit functions for this thread
            auto* p = get_self_id_data();
            if (HPX_LIKELY(p != nullptr))
            {
                p->run_thread_exit_callbacks();
                p->free_thread_exit_callbacks();
            }

            return threads::thread_result_type(
                threads::thread_schedule_state::terminated,
                threads::invalid_thread_id);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    threads::thread_id_ref_type register_thread(threads::thread_init_data& data,
        threads::thread_pool_base* pool, error_code& ec)
    {
        HPX_ASSERT(pool);

        threads::thread_id_ref_type id = threads::invalid_thread_id;
        data.run_now = true;
        pool->create_thread(data, id, ec);
        return id;
    }

    void register_thread(threads::thread_init_data& data,
        threads::thread_pool_base* pool, threads::thread_id_ref_type& id,
        error_code& ec)
    {
        HPX_ASSERT(pool);

        data.run_now = true;
        pool->create_thread(data, id, ec);
    }

    void register_thread(threads::thread_init_data& data,
        threads::thread_id_ref_type& id, error_code& ec)
    {
        auto* pool = detail::get_self_or_default_pool();
        HPX_ASSERT(pool);

        data.run_now = true;
        pool->create_thread(data, id, ec);
    }

    threads::thread_id_ref_type register_thread(
        threads::thread_init_data& data, error_code& ec)
    {
        auto* pool = detail::get_self_or_default_pool();
        HPX_ASSERT(pool);

        threads::thread_id_ref_type id = threads::invalid_thread_id;
        data.run_now = true;
        pool->create_thread(data, id, ec);
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    thread_id_ref_type register_work(threads::thread_init_data& data,
        threads::thread_pool_base* pool, error_code& ec)
    {
        HPX_ASSERT(pool);
        data.run_now = false;
        return pool->create_work(data, ec);
    }

    thread_id_ref_type register_work(
        threads::thread_init_data& data, error_code& ec)
    {
        auto* pool = detail::get_self_or_default_pool();
        HPX_ASSERT(pool);

        data.run_now = false;
        return pool->create_work(data, ec);
    }
}    // namespace hpx::threads
