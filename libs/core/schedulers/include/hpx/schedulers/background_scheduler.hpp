//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/schedulers/deadlock_detection.hpp>
#include <hpx/schedulers/local_queue_scheduler.hpp>
#include <hpx/schedulers/lockfree_queue_backends.hpp>
#include <hpx/schedulers/thread_queue.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::threads::policies {

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
    using default_static_queue_scheduler_terminated_queue = lockfree_lifo;
#else
    using default_static_queue_scheduler_terminated_queue = lockfree_fifo;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // The background_scheduler_scheduler runs only background work
    template <typename Mutex = std::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing =
            default_static_queue_scheduler_terminated_queue>
    class background_scheduler
      : public local_queue_scheduler<Mutex, PendingQueuing, StagedQueuing,
            TerminatedQueuing>
    {
    public:
        using base_type = local_queue_scheduler<Mutex, PendingQueuing,
            StagedQueuing, TerminatedQueuing>;

        explicit background_scheduler(
            typename base_type::init_parameter_type const& init,
            bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {
        }

        static std::string get_scheduler_name()
        {
            return "background_scheduler";
        }

        void set_scheduler_mode(scheduler_mode mode) override
        {
            // this scheduler does not support stealing or numa stealing, but
            // needs to enable background work
            mode = scheduler_mode(mode & ~scheduler_mode::enable_stealing);
            mode = scheduler_mode(mode & ~scheduler_mode::enable_stealing_numa);
            mode = scheduler_mode(mode | ~scheduler_mode::do_background_work);
            mode =
                scheduler_mode(mode | ~scheduler_mode::do_background_work_only);
            scheduler_base::set_scheduler_mode(mode);
        }

        // Return the next thread to be executed, return false if none is
        // available. Note that this scheduler can't be used for any real work,
        // only background work is processed by the scheduling loop
        bool get_next_thread(
            std::size_t, bool, threads::thread_id_ref_type&, bool) override
        {
            return false;
        }

        // This is a function which gets called periodically by the thread
        // manager to allow for maintenance tasks to be executed in the
        // scheduler. Returns true if the OS thread calling this function has to
        // be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t, bool running, std::int64_t&, bool,
            std::size_t&) override
        {
            return !running;
        }

        void create_thread(
            thread_init_data&, thread_id_ref_type*, error_code&) override
        {
            HPX_THROW_EXCEPTION(error::bad_function_call,
                "background_scheduler::create_thread",
                "This scheduler does not support managing 'normal' threads");
        }

        void schedule_thread(threads::thread_id_ref_type,
            threads::thread_schedule_hint, bool,
            thread_priority = thread_priority::default_) override
        {
            HPX_THROW_EXCEPTION(error::bad_function_call,
                "background_scheduler::schedule_thread",
                "This scheduler does not support managing 'normal' threads");
        }

        void schedule_thread_last(threads::thread_id_ref_type,
            threads::thread_schedule_hint, bool,
            thread_priority = thread_priority::default_) override
        {
            HPX_THROW_EXCEPTION(error::bad_function_call,
                "background_scheduler::schedule_thread_last",
                "This scheduler does not support managing 'normal' threads");
        }

        void destroy_thread(threads::thread_data*) override
        {
            HPX_THROW_EXCEPTION(error::bad_function_call,
                "background_scheduler::destroy_thread",
                "This scheduler does not support managing 'normal' threads");
        }
    };
}    // namespace hpx::threads::policies

#include <hpx/config/warnings_suffix.hpp>
