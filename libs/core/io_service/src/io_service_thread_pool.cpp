//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/affinity/affinity_data.hpp>
#include <hpx/assert.hpp>
#include <hpx/concurrency/barrier.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/io_service/io_service_thread_pool.hpp>
#include <hpx/threading_base/callback_notifier.hpp>
#include <hpx/threading_base/scheduler_mode.hpp>

#include <cstddef>
#include <exception>

namespace hpx { namespace threads { namespace detail {
    io_service_thread_pool::io_service_thread_pool(
        hpx::threads::thread_pool_init_parameters const& init)
      : thread_pool_base(init)
      , threads_(init.notifier_, init.name_.c_str())
    {
    }

    hpx::state io_service_thread_pool::get_state() const
    {
        return state_stopped;
    }

    hpx::state io_service_thread_pool::get_state(
        std::size_t /* num_thread */) const
    {
        return state_stopped;
    }

    bool io_service_thread_pool::has_reached_state(hpx::state /* s */) const
    {
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    void io_service_thread_pool::create_thread(thread_init_data& /* data */,
        thread_id_type& /* id */, error_code& /* ec */)
    {
    }

    void io_service_thread_pool::create_work(
        thread_init_data& /* data */, error_code& /* ec */)
    {
    }

    threads::thread_state io_service_thread_pool::set_state(
        thread_id_type const& /* id */, thread_schedule_state /* new_state */,
        thread_restart_state /* new_state_ex */, thread_priority /* priority */,
        error_code& /* ec */)
    {
        return threads::thread_state(threads::thread_schedule_state::terminated,
            threads::thread_restart_state::unknown);
    }

    threads::thread_id_type io_service_thread_pool::set_state(
        hpx::chrono::steady_time_point const& /* abs_time */,
        thread_id_type const& id, thread_schedule_state /* newstate */,
        thread_restart_state /* newstate_ex */, thread_priority /* priority */,
        error_code& /* ec */)
    {
        return id;
    }

    void io_service_thread_pool::report_error(
        std::size_t /* num */, std::exception_ptr const& /* e */)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    bool io_service_thread_pool::run(
        std::unique_lock<std::mutex>& l, std::size_t num_threads)
    {
        HPX_ASSERT(l.owns_lock());
        HPX_UNUSED(l);
        util::barrier startup(1);
        return threads_.run(num_threads, false, &startup);
    }

    void io_service_thread_pool::stop(
        std::unique_lock<std::mutex>& /* l */, bool /* blocking = true */)
    {
    }

    void io_service_thread_pool::resume_direct(error_code& /* ec */)
    {
        HPX_ASSERT_MSG(
            false, "Resuming io_service_thread_pool is not supported");
    }

    void io_service_thread_pool::suspend_direct(error_code& /* ec */)
    {
        HPX_ASSERT_MSG(
            false, "Suspending io_service_thread_pool is not supported");
    }

    void io_service_thread_pool::suspend_processing_unit_direct(
        std::size_t /* virt_core */, error_code& /* ec */)
    {
        HPX_ASSERT_MSG(false,
            "Suspending threads on io_service_thread_pool is not "
            "supported");
    }

    void io_service_thread_pool::resume_processing_unit_direct(
        std::size_t /* virt_core */, error_code& /* ec */)
    {
        HPX_ASSERT_MSG(false,
            "Suspending threads on io_service_thread_pool is not "
            "supported");
    }

    std::thread& io_service_thread_pool::get_os_thread_handle(
        std::size_t global_thread_num)
    {
        return threads_.get_os_thread_handle(
            global_thread_num - this->thread_offset_);
    }

    std::size_t io_service_thread_pool::get_os_thread_count() const
    {
        return threads_.size();
    }

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
    ///////////////////////////////////////////////////////////////////////////
    // detail::manage_executor implementation
    std::size_t io_service_thread_pool::get_policy_element(
        executor_parameter /* p */, error_code& /* ec */) const
    {
        return 0;
    }

    void io_service_thread_pool::get_statistics(
        executor_statistics& /* stats */, error_code& /* ec */) const
    {
    }

    void io_service_thread_pool::add_processing_unit(
        std::size_t /* virt_core */, std::size_t /* thread_num */,
        error_code& /* ec */)
    {
    }

    void io_service_thread_pool::remove_processing_unit(
        std::size_t /* thread_num */, error_code& /* ec */)
    {
    }
#endif
}}}    // namespace hpx::threads::detail
