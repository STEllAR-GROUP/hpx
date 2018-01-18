//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/threads/detail/io_service_thread_pool.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <exception>

namespace hpx { namespace threads { namespace detail
{
    io_service_thread_pool::io_service_thread_pool(
            threads::policies::callback_notifier& notifier, std::size_t index,
            char const* pool_name, policies::scheduler_mode m,
            std::size_t thread_offset)
      : thread_pool_base(notifier, index, pool_name, m, thread_offset)
      , threads_(notifier.on_start_thread_, notifier.on_stop_thread_, pool_name)
    {
    }

    hpx::state io_service_thread_pool::get_state() const
    {
        return state_stopped;
    }

    hpx::state io_service_thread_pool::get_state(std::size_t num_thread) const
    {
        return state_stopped;
    }

    bool io_service_thread_pool::has_reached_state(hpx::state s) const
    {
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    void io_service_thread_pool::create_thread(thread_init_data& data,
        thread_id_type& id, thread_state_enum initial_state, bool run_now,
        error_code& ec)
    {
    }

    void io_service_thread_pool::create_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
    }

    threads::thread_state io_service_thread_pool::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return threads::thread_state(threads::terminated, threads::wait_unknown);
    }

    threads::thread_id_type io_service_thread_pool::set_state(
        util::steady_time_point const& abs_time, thread_id_type const& id,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, error_code& ec)
    {
        return id;
    }

    void io_service_thread_pool::report_error(
        std::size_t num, std::exception_ptr const& e)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    bool io_service_thread_pool::run(std::unique_lock<compat::mutex>& l,
        std::size_t num_threads)
    {
        HPX_ASSERT(l.owns_lock());
        compat::barrier startup(1);
        return threads_.run(num_threads, false, &startup);
    }

    void io_service_thread_pool::stop(
        std::unique_lock<compat::mutex>& l, bool blocking /*= true*/)
    {
    }

    hpx::compat::thread& io_service_thread_pool::get_os_thread_handle(
        std::size_t global_thread_num)
    {
        return threads_.get_os_thread_handle(
            global_thread_num - this->thread_offset_);
    }

    std::size_t io_service_thread_pool::get_os_thread_count() const
    {
        return threads_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    // detail::manage_executor implementation
    std::size_t io_service_thread_pool::get_policy_element(
        executor_parameter p, error_code& ec) const
    {
        return 0;
    }

    void io_service_thread_pool::get_statistics(
        executor_statistics& stats, error_code& ec) const
    {
    }

    void io_service_thread_pool::add_processing_unit(
        std::size_t virt_core, std::size_t thread_num, error_code& ec)
    {
    }

    void io_service_thread_pool::remove_processing_unit(
        std::size_t thread_num, error_code& ec)
    {
    }
}}}
