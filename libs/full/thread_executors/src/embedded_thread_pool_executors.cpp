//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/thread_executors/embedded_thread_pool_executors.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY) &&                        \
    defined(HPX_HAVE_EMBEDDED_THREAD_POOLS_COMPATIBILITY)

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/schedulers/local_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/schedulers/static_queue_scheduler.hpp>
#endif
#include <hpx/schedulers/local_priority_queue_scheduler.hpp>
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/schedulers/static_priority_queue_scheduler.hpp>
#endif
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/thread_executors/detail/on_self_reset.hpp>
#include <hpx/thread_executors/manage_thread_executor.hpp>
#include <hpx/thread_executors/resource_manager.hpp>
#include <hpx/thread_pools/scheduling_loop.hpp>
#include <hpx/threading_base/create_thread.hpp>
#include <hpx/threading_base/set_thread_state.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

namespace hpx { namespace threads { namespace executors { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    embedded_thread_pool_executor<Scheduler>::embedded_thread_pool_executor(
        std::size_t max_punits, std::size_t min_punits, char const* description,
        policies::detail::affinity_data const& affinity_data)
      : scheduler_(typename Scheduler::init_parameter_type(
                       max_punits, affinity_data, description),
            false)
      , shutdown_sem_(0)
      , current_concurrency_(0)
      , max_current_concurrency_(0)
      , tasks_scheduled_(0)
      , tasks_completed_(0)
      , max_punits_(max_punits)
      , min_punits_(min_punits)
      , curr_punits_(0)
      , cookie_(0)
      , self_(max_punits)
      , affinity_data_(affinity_data)
    {
        scheduler_.add_scheduler_mode(
            policies::scheduler_mode::enable_stealing_numa);
        if (max_punits < min_punits)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "embedded_thread_pool_executor<Scheduler>::"
                "embedded_thread_pool_executor",
                "max_punit shouldn't be smaller than min_punit");
            return;
        }
        if (max_punits >
            hpx::parallel::execution::detail::get_os_thread_count())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "embedded_thread_pool_executor<Scheduler>::"
                "embedded_thread_pool_executor",
                "max_punit shouldn't be larger than number of available "
                "OS-threads");
            return;
        }

        scheduler_.set_parent_pool(this_thread::get_pool());

        // Inform the resource manager about this new executor. This causes the
        // resource manager to interact with this executor using the
        // manage_executor interface.
        resource_manager& rm = resource_manager::get();
        cookie_ = rm.initial_allocation(
            new manage_thread_executor<embedded_thread_pool_executor>(*this));
    }

    template <typename Scheduler>
    embedded_thread_pool_executor<Scheduler>::~embedded_thread_pool_executor()
    {
        // if we're still starting up, give this executor a chance of executing
        // its tasks
        hpx::util::yield_while(
            [this]() { return scheduler_.get_state(0) < state_running; },
            "this_thread_executor<Scheduler>::~this_thread_executor()");

        // Wait for work to finish.
        hpx::util::yield_while(
            [this]() {
                return scheduler_.get_thread_count() >
                    scheduler_.get_background_thread_count();
            },
            "this_thread_executor<Scheduler>::~this_thread_executor()");

        // Inform the resource manager that this executor is about to be
        // destroyed. This will cause it to invoke remove_processing_unit below
        // for each of the currently allocated virtual cores.
        resource_manager& rm = resource_manager::get();
        rm.stop_executor(cookie_);

        // wait for executor to finish executing
        shutdown_sem_.wait(max_current_concurrency_.load());

        // detach this executor from resource manager
        rm.detach(cookie_);    // this releases proxy (manage_thread_executor)

#if defined(HPX_DEBUG)
        // all resources should have been stopped at this point (or have never
        // been initialized)
        for (std::size_t i = 0; i != max_punits_; ++i)
        {
            hpx::state s = scheduler_.get_state(i).load();
            HPX_ASSERT(s == state_initialized || s == state_stopped);
        }

        // all scheduled tasks should have completed executing
        HPX_ASSERT(tasks_completed_ == tasks_scheduled_);

        // all driver threads should have stopped executing
        HPX_ASSERT(current_concurrency_ == 0);
#endif
    }

    template <typename Scheduler>
    threads::thread_result_type
    embedded_thread_pool_executor<Scheduler>::thread_function_nullary(
        closure_type func)
    {
        // execute the actual thread function
        func();

        // update statistics
        ++tasks_completed_;

        // Verify that there are no more registered locks for this
        // OS-thread. This will throw if there are still any locks
        // held.
        util::force_error_on_lock();

        return threads::thread_result_type(
            threads::thread_schedule_state::terminated,
            threads::invalid_thread_id);
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::add(closure_type&& f,
        util::thread_description const& desc,
        threads::thread_schedule_state initial_state, bool run_now,
        threads::thread_stacksize stacksize,
        threads::thread_schedule_hint /* schedulehint */, error_code& ec)
    {
        // create a new thread
        thread_init_data data(
            util::one_shot(util::bind(
                &embedded_thread_pool_executor::thread_function_nullary, this,
                std::move(f))),
            desc, thread_priority::default_, thread_schedule_hint(), stacksize,
            initial_state, run_now);

        // update statistics
        ++tasks_scheduled_;

        threads::thread_id_type id = threads::invalid_thread_id;
        threads::detail::create_thread(&scheduler_, data, id, ec);
        if (ec)
        {
            --tasks_scheduled_;
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::add_at(
        std::chrono::steady_clock::time_point const& abs_time, closure_type&& f,
        util::thread_description const& desc,
        threads::thread_stacksize /* stacksize */, error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(
            util::one_shot(util::bind(
                &embedded_thread_pool_executor::thread_function_nullary, this,
                std::move(f))),
            desc, thread_priority::default_, thread_schedule_hint(),
            thread_stacksize::default_, thread_schedule_state::suspended, true);

        threads::thread_id_type id = threads::invalid_thread_id;
        threads::detail::create_thread(    //-V601
            &scheduler_, data, id, ec);
        if (ec)
            return;
        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // update statistics
        ++tasks_scheduled_;

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(
            scheduler_, abs_time, id, nullptr, true, ec);
        if (ec)
        {
            --tasks_scheduled_;
            return;
        }

        if (&ec != &throws)
            ec = make_success_code();
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::add_after(
        std::chrono::steady_clock::duration const& rel_time, closure_type&& f,
        util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(std::chrono::steady_clock::now() + rel_time, std::move(f),
            desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    std::uint64_t
    embedded_thread_pool_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();
        return scheduler_.get_queue_length();
    }

    // Reset internal (round robin) thread distribution scheme
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::reset_thread_distribution()
    {
        scheduler_.reset_thread_distribution();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void
    embedded_thread_pool_executor<Scheduler>::suspend_back_into_calling_context(
        std::size_t virt_core)
    {
        // give invoking context a chance to catch up with its tasks, but only
        // if this scheduler is currently in running state
        std::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_running;
        if (state.compare_exchange_strong(expected, state_suspended))
        {
            {
                on_self_reset on_exit(self_[virt_core]);
                hpx::execution_base::this_thread::yield();
            }

            // reset state to running if current state is still suspended
            expected = state_suspended;
            state.compare_exchange_strong(expected, state_running);
        }
        else
        {
            HPX_ASSERT(expected != state_suspended);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct on_run_exit
    {
        on_run_exit(std::atomic<std::size_t>& current_concurrency,
            lcos::local::counting_semaphore& shutdown_sem,
            threads::thread_self* self)
          : current_concurrency_(current_concurrency)
          , shutdown_sem_(shutdown_sem)
          , self_(self)
        {
            threads::detail::set_self_ptr(nullptr);
            ++current_concurrency_;
        }

        ~on_run_exit()
        {
            --current_concurrency_;
            threads::detail::set_self_ptr(self_);
            shutdown_sem_.signal();
        }

        std::atomic<std::size_t>& current_concurrency_;
        lcos::local::counting_semaphore& shutdown_sem_;
        threads::thread_self* self_;
    };

    // execute all work
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::run(
        std::size_t virt_core, std::size_t /* thread_num */)
    {
        // Set the state to 'state_running' only if it's still in 'state_starting'
        // state, otherwise our destructor is currently being executed, which
        // means we need to still execute all threads.
        std::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_starting;
        if (state.compare_exchange_strong(expected, state_running))
        {
            ++max_current_concurrency_;

            self_[virt_core] = threads::get_self_ptr();

            on_run_exit on_exit(
                current_concurrency_, shutdown_sem_, self_[virt_core]);

            // FIXME: turn these values into performance counters
            std::int64_t executed_threads = 0, executed_thread_phases = 0;
            std::int64_t overall_times = 0, thread_times = 0;
            std::int64_t idle_loop_count = 0, busy_loop_count = 0;
            bool task_active = false;

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
            std::int64_t bg_work = 0;
            std::int64_t bg_send = 0;
            std::int64_t bg_receive = 0;
            threads::detail::scheduling_counters counters(executed_threads,
                executed_thread_phases, overall_times, thread_times,
                idle_loop_count, busy_loop_count, task_active, bg_work, bg_send,
                bg_receive);
#else
            threads::detail::scheduling_counters counters(executed_threads,
                executed_thread_phases, overall_times, thread_times,
                idle_loop_count, busy_loop_count, task_active);
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

            threads::detail::scheduling_callbacks callbacks(nullptr,
                util::deferred_call(    //-V107
                    &embedded_thread_pool_executor::
                        suspend_back_into_calling_context,
                    this, virt_core));

            scheduler_.add_scheduler_mode(policies::fast_idle_mode);
            threads::detail::scheduling_loop(
                virt_core, scheduler_, counters, callbacks);

            // the scheduling_loop is allowed to exit only if no more HPX
            // threads exist
            HPX_ASSERT(
                (scheduler_.get_thread_count(thread_schedule_state::suspended,
                     thread_priority::default_, virt_core) == 0 &&
                    scheduler_.get_queue_length(virt_core) == 0) ||
                state >= state_terminating);
        }
    }

    // Return statistics collected by this scheduler
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::get_statistics(
        executor_statistics& stats, error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        stats.queue_length_ = scheduler_.get_queue_length();
        stats.tasks_scheduled_ = tasks_scheduled_.load();
        stats.tasks_completed_ = tasks_completed_.load();
    }

    template <typename Scheduler>
    char const* embedded_thread_pool_executor<Scheduler>::get_description()
        const
    {
        return scheduler_.get_description();
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t embedded_thread_pool_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch (p)
        {
        case threads::detail::min_concurrency:
            return min_punits_;

        case threads::detail::max_concurrency:
            return max_punits_;

        case threads::detail::current_concurrency:
            return curr_punits_;

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "embedded_thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    // Provide the given processing unit to the scheduler.
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::add_processing_unit(
        std::size_t virt_core, std::size_t thread_num, error_code& ec)
    {
        std::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_initialized;
        if (state.compare_exchange_strong(expected, state_starting))
        {
            ++curr_punits_;
            thread_init_data data(
                make_thread_function_nullary(
                    util::deferred_call(&embedded_thread_pool_executor::run,
                        this, virt_core, thread_num)),
                "embedded_thread_pool_executor thread",
                threads::thread_priority::normal,
                threads::thread_schedule_hint(
                    static_cast<std::int16_t>(thread_num)),
                threads::thread_stacksize::default_,
                threads::thread_schedule_state::pending, true);
            register_thread(data, ec);
        }
    }

    // Remove the given processing unit from the scheduler.
    template <typename Scheduler>
    void embedded_thread_pool_executor<Scheduler>::remove_processing_unit(
        std::size_t virt_core, error_code& /* ec */)
    {
        // inform the scheduler to stop the virtual core
        std::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state oldstate = state.exchange(state_stopped);
        HPX_ASSERT(oldstate == state_starting || oldstate == state_running ||
            oldstate == state_suspended || oldstate == state_stopped);
        HPX_UNUSED(oldstate);
        --curr_punits_;
    }
}}}}    // namespace hpx::threads::executors::detail

namespace hpx { namespace threads { namespace executors {
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    local_queue_executor::local_queue_executor()
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::local_queue_scheduler<>>(
            parallel::execution::detail::get_os_thread_count(), 1,
            "local_queue_executor"))
    {
    }

    local_queue_executor::local_queue_executor(
        std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::local_queue_scheduler<>>(
            max_punits, min_punits, "local_queue_executor"))
    {
    }
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_queue_executor::static_queue_executor()
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::static_queue_scheduler<>>(
            parallel::execution::detail::get_os_thread_count(), 1,
            "static_queue_executor"))
    {
    }

    static_queue_executor::static_queue_executor(
        std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::static_queue_scheduler<>>(
            max_punits, min_punits, "static_queue_executor"))
    {
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_executor::local_priority_queue_executor()
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::local_priority_queue_scheduler<>>(
            parallel::execution::detail::get_os_thread_count(), 1,
            "local_priority_queue_executor"))
    {
    }

    local_priority_queue_executor::local_priority_queue_executor(
        std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::local_priority_queue_scheduler<>>(
            max_punits, min_punits, "local_priority_queue_executor"))
    {
    }

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_executor::static_priority_queue_executor()
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::static_priority_queue_scheduler<>>(
            parallel::execution::detail::get_os_thread_count(), 1,
            "static_priority_queue_executor"))
    {
    }

    static_priority_queue_executor::static_priority_queue_executor(
        std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::embedded_thread_pool_executor<
            policies::static_priority_queue_scheduler<>>(
            max_punits, min_punits, "static_priority_queue_executor"))
    {
    }
#endif

}}}    // namespace hpx::threads::executors
#endif
