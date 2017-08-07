//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_IO_SERVICE_THREAD_POOL_HPP)
#define HPX_IO_SERVICE_THREAD_POOL_HPP

#include <hpx/config.hpp>
#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/state.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT io_service_thread_pool : public thread_pool_base
    {
    public:
        io_service_thread_pool(threads::policies::callback_notifier& notifier,
            std::size_t index, char const* pool_name,
            policies::scheduler_mode m = policies::scheduler_mode::nothing_special,
            std::size_t thread_offset = 0);

        void print_pool(std::ostream& os) {}

        ///////////////////////////////////////////////////////////////////////
        hpx::state get_state() const;
        hpx::state get_state(std::size_t num_thread) const;
        bool has_reached_state(hpx::state s) const;

        ///////////////////////////////////////////////////////////////////////
        void create_thread(thread_init_data& data, thread_id_type& id,
            thread_state_enum initial_state, bool run_now, error_code& ec);

        void create_work(thread_init_data& data,
            thread_state_enum initial_state, error_code& ec);

        thread_state set_state(thread_id_type const& id,
            thread_state_enum new_state, thread_state_ex_enum new_state_ex,
            thread_priority priority, error_code& ec);

        thread_id_type set_state(util::steady_time_point const& abs_time,
            thread_id_type const& id, thread_state_enum newstate,
            thread_state_ex_enum newstate_ex, thread_priority priority,
            error_code& ec);

        void report_error(std::size_t num, std::exception_ptr const& e);

        ///////////////////////////////////////////////////////////////////////
        bool run(std::unique_lock<compat::mutex>& l, std::size_t pool_threads);

        ///////////////////////////////////////////////////////////////////////
        void stop (std::unique_lock<compat::mutex>& l, bool blocking = true);

        ///////////////////////////////////////////////////////////////////////
        compat::thread& get_os_thread_handle(std::size_t global_thread_num);

        std::size_t get_os_thread_count() const;

        ///////////////////////////////////////////////////////////////////////
        // detail::manage_executor implementation

        // Return the requested policy element
        std::size_t get_policy_element(
            executor_parameter p, error_code& ec) const;

        // Return statistics collected by this scheduler
        void get_statistics(executor_statistics& stats, error_code& ec) const;

        // Provide the given processing unit to the scheduler.
        void add_processing_unit(std::size_t virt_core, std::size_t thread_num,
            error_code& ec);

        // Remove the given processing unit from the scheduler.
        void remove_processing_unit(std::size_t thread_num, error_code& ec);

    private:
        util::io_service_pool threads_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
