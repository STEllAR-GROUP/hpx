//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/threading_base/callback_notifier.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>
#include <hpx/threading_base/scheduler_state.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <mutex>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT io_service_thread_pool : public thread_pool_base
    {
    public:
        explicit io_service_thread_pool(
            threads::thread_pool_init_parameters const& init);

        void print_pool(std::ostream&) {}

        ///////////////////////////////////////////////////////////////////////
        hpx::state get_state() const;
        hpx::state get_state(std::size_t num_thread) const;
        bool has_reached_state(hpx::state s) const;

        ///////////////////////////////////////////////////////////////////////
        void create_thread(
            thread_init_data& data, thread_id_type& id, error_code& ec);

        void create_work(thread_init_data& data, error_code& ec);

        thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec);

        thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec);

        void report_error(std::size_t num, std::exception_ptr const& e);

        ///////////////////////////////////////////////////////////////////////
        bool run(std::unique_lock<std::mutex>& l, std::size_t pool_threads);

        ///////////////////////////////////////////////////////////////////////
        void stop(std::unique_lock<std::mutex>& l, bool blocking = true);

        ///////////////////////////////////////////////////////////////////////
        void resume_direct(error_code& ec = throws);

        void suspend_direct(error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        void suspend_processing_unit_direct(
            std::size_t virt_core, error_code& ec = throws);

        void resume_processing_unit_direct(
            std::size_t virt_core, error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        std::thread& get_os_thread_handle(std::size_t global_thread_num);

        std::size_t get_os_thread_count() const;

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
        ///////////////////////////////////////////////////////////////////////
        // detail::manage_executor implementation

        // Return the requested policy element
        std::size_t get_policy_element(
            executor_parameter p, error_code& ec) const;

        // Return statistics collected by this scheduler
        void get_statistics(executor_statistics& stats, error_code& ec) const;

        // Provide the given processing unit to the scheduler.
        void add_processing_unit(
            std::size_t virt_core, std::size_t thread_num, error_code& ec);

        // Remove the given processing unit from the scheduler.
        void remove_processing_unit(std::size_t thread_num, error_code& ec);
#endif

    private:
        util::io_service_pool threads_;
    };
}}}    // namespace hpx::threads::detail

#include <hpx/config/warnings_suffix.hpp>
