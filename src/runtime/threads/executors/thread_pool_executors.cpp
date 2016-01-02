//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/resource_manager.hpp>
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
#endif
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
#if defined(HPX_HAVE_THROTTLE_SCHEDULER)
#include <hpx/runtime/threads/policies/throttle_queue_scheduler.hpp>
#endif
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
#endif
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/executors/thread_pool_executors.hpp>
#include <hpx/runtime/threads/executors/manage_thread_executor.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace threads { namespace detail
{
    // The function \a set_self_ptr sets a pointer to the (OS thread
    // specific) self reference to the current HPX thread.
    void set_self_ptr(threads::thread_self*);
}}}

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool_executor<Scheduler>::thread_pool_executor(
            std::size_t max_punits, std::size_t min_punits,
            char const* description)
      : scheduler_(
            typename Scheduler::init_parameter_type(max_punits, description),
            false
        ),
        shutdown_sem_(0),
        current_concurrency_(0), max_current_concurrency_(0),
        tasks_scheduled_(0), tasks_completed_(0),
        max_punits_(max_punits), min_punits_(min_punits), curr_punits_(0),
        cookie_(0),
        self_(max_punits)
    {
        if (max_punits < min_punits)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "thread_pool_executor<Scheduler>::thread_pool_executor",
                "max_punit shouldn't be smaller than min_punit");
            return;
        }
        if (max_punits > hpx::get_os_thread_count())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "thread_pool_executor<Scheduler>::thread_pool_executor",
                "max_punit shouldn't be larger than number of available "
                "OS-threads");
            return;
        }

        // Inform the resource manager about this new executor. This causes the
        // resource manager to interact with this executor using the
        // manage_executor interface.
        resource_manager& rm = resource_manager::get();
        cookie_ = rm.initial_allocation(
            new manage_thread_executor<thread_pool_executor>(*this));
    }

    template <typename Scheduler>
    thread_pool_executor<Scheduler>::~thread_pool_executor()
    {
        // if we're still starting up, give this executor a chance of executing
        // its tasks
        while (!scheduler_.has_reached_state(state_running))
            this_thread::suspend();

        // Inform the resource manager that this executor is about to be
        // destroyed. This will cause it to invoke remove_processing_unit below
        // for each of the currently allocated virtual cores.
        resource_manager& rm = resource_manager::get();
        rm.stop_executor(cookie_);

        // wait for executor to finish executing
        shutdown_sem_.wait(max_current_concurrency_.load());

        // detach this executor from resource manager
        rm.detach(cookie_);     // this releases proxy (manage_thread_executor)

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
    threads::thread_state_enum
    thread_pool_executor<Scheduler>::thread_function_nullary(
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

        return threads::terminated;
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add(
        closure_type && f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, threads::thread_stacksize stacksize, error_code& ec)
    {
        // create a new thread
        thread_init_data data(util::bind(
            util::one_shot(&thread_pool_executor::thread_function_nullary),
            this, std::move(f)), desc);
        data.stacksize = threads::get_stack_size(stacksize);

        // update statistics
        ++tasks_scheduled_;

        threads::thread_id_type id = threads::invalid_thread_id;
        threads::detail::create_thread(&scheduler_, data, id, initial_state, //-V601
            run_now, ec);
        if (ec) {
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
    void thread_pool_executor<Scheduler>::add_at(
        boost::chrono::steady_clock::time_point const& abs_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        // create a new suspended thread
        thread_init_data data(util::bind(
            util::one_shot(&thread_pool_executor::thread_function_nullary),
            this, std::move(f)), desc);
        data.stacksize = threads::get_stack_size(stacksize);

        threads::thread_id_type id = threads::invalid_thread_id;
        threads::detail::create_thread( //-V601
            &scheduler_, data, id, suspended, true, ec);
        if (ec) return;
        HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

        // update statistics
        ++tasks_scheduled_;

        // now schedule new thread for execution
        threads::detail::set_thread_state_timed(scheduler_, abs_time, id, ec);
        if (ec) {
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
    void thread_pool_executor<Scheduler>::add_after(
        boost::chrono::steady_clock::duration const& rel_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(boost::chrono::steady_clock::now() + rel_time,
            std::move(f), desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    template <typename Scheduler>
    boost::uint64_t thread_pool_executor<Scheduler>::num_pending_closures(
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();
        return scheduler_.get_queue_length();
    }


    // Reset internal (round robin) thread distribution scheme
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::reset_thread_distribution()
    {
        scheduler_.reset_thread_distribution();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct on_self_reset
    {
        on_self_reset(threads::thread_self* self)
        {
            threads::detail::set_self_ptr(self);
        }
        ~on_self_reset()
        {
            threads::detail::set_self_ptr(0);
        }
    };

    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::suspend_back_into_calling_context(
        std::size_t virt_core)
    {
        // give invoking context a chance to catch up with its tasks, but only
        // if this scheduler is currently in running state
        boost::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_running;
        if (state.compare_exchange_strong(expected, state_suspended))
        {
            {
                on_self_reset on_exit(self_[virt_core]);
                this_thread::suspend();
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
        on_run_exit(boost::atomic<std::size_t>& current_concurrency,
                lcos::local::counting_semaphore& shutdown_sem,
                threads::thread_self* self)
          : current_concurrency_(current_concurrency),
            shutdown_sem_(shutdown_sem),
            self_(self)
        {
            threads::detail::set_self_ptr(0);
            ++current_concurrency_;
        }

        ~on_run_exit()
        {
            --current_concurrency_;
            threads::detail::set_self_ptr(self_);
            shutdown_sem_.signal();
        }

        boost::atomic<std::size_t>& current_concurrency_;
        lcos::local::counting_semaphore& shutdown_sem_;
        threads::thread_self* self_;
    };

    // execute all work
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::run(std::size_t virt_core,
        std::size_t thread_num)
    {
        // Set the state to 'state_running' only if it's still in 'state_starting'
        // state, otherwise our destructor is currently being executed, which
        // means we need to still execute all threads.
        boost::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_starting;
        if (state.compare_exchange_strong(expected, state_running))
        {
            ++max_current_concurrency_;

            {
                boost::lock_guard<mutex_type> l(mtx_);
                scheduler_.add_punit(virt_core, thread_num);
                scheduler_.on_start_thread(virt_core);
            }

            self_[virt_core] = threads::get_self_ptr();

            on_run_exit on_exit(current_concurrency_, shutdown_sem_,
                self_[virt_core]);

            // FIXME: turn these values into performance counters
            boost::int64_t executed_threads = 0, executed_thread_phases = 0;
            boost::uint64_t overall_times = 0, thread_times = 0;

            threads::detail::scheduling_counters counters(
                executed_threads, executed_thread_phases,
                overall_times, thread_times);

            threads::detail::scheduling_callbacks callbacks(
                threads::detail::scheduling_callbacks::callback_type(),
                util::bind( //-V107
                    &thread_pool_executor::suspend_back_into_calling_context,
                    this, virt_core));

            scheduler_.set_scheduler_mode(policies::fast_idle_mode);
            threads::detail::scheduling_loop(virt_core, scheduler_,
                counters, callbacks);

            // the scheduling_loop is allowed to exit only if no more HPX
            // threads exist
            HPX_ASSERT(!scheduler_.get_thread_count(
                unknown, thread_priority_default, virt_core) ||
                state == state_terminating);
        }
    }

    // Return statistics collected by this scheduler
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::get_statistics(
        executor_statistics& stats, error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        stats.queue_length_ = scheduler_.get_queue_length();
        stats.tasks_scheduled_ = tasks_scheduled_.load();
        stats.tasks_completed_ = tasks_completed_.load();
    }

    // Return the requested policy element
    template <typename Scheduler>
    std::size_t thread_pool_executor<Scheduler>::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
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
            "thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    // Provide the given processing unit to the scheduler.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::add_processing_unit(
        std::size_t virt_core, std::size_t thread_num, error_code& ec)
    {
        boost::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state expected = state_initialized;
        if (state.compare_exchange_strong(expected, state_starting))
        {
            ++curr_punits_;
            register_thread_nullary(
                util::bind(
                    util::one_shot(&thread_pool_executor::run),
                    this, virt_core, thread_num
                ),
                "thread_pool_executor thread", threads::pending, true,
                threads::thread_priority_normal, thread_num,
                threads::thread_stacksize_default, ec);
        }
    }

    // Remove the given processing unit from the scheduler.
    template <typename Scheduler>
    void thread_pool_executor<Scheduler>::remove_processing_unit(
        std::size_t virt_core, error_code& ec)
    {
        // inform the scheduler to stop the virtual core
        boost::atomic<hpx::state>& state = scheduler_.get_state(virt_core);
        hpx::state oldstate = state.exchange(state_stopped);
        HPX_ASSERT(oldstate == state_running || oldstate == state_suspended ||
            oldstate == state_stopped);
        --curr_punits_;
    }
}}}}

namespace hpx { namespace threads { namespace executors
{
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    local_queue_executor::local_queue_executor()
      : scheduled_executor(new detail::thread_pool_executor<
            policies::local_queue_scheduler<> >(
                get_os_thread_count(), 1, "local_queue_executor"))
    {}

    local_queue_executor::local_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::thread_pool_executor<
            policies::local_queue_scheduler<> >(
                max_punits, min_punits, "local_queue_executor"))
    {}
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_queue_executor::static_queue_executor()
      : scheduled_executor(new detail::thread_pool_executor<
            policies::static_queue_scheduler<> >(
                get_os_thread_count(), 1, "static_queue_executor"))
    {}

    static_queue_executor::static_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::thread_pool_executor<
            policies::static_queue_scheduler<> >(
                max_punits, min_punits, "static_queue_executor"))
    {}
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    throttle_queue_executor::throttle_queue_executor()
      : scheduled_executor(new detail::thread_pool_executor<
            policies::throttle_queue_scheduler<> >(
                get_os_thread_count(), 1))
    {}

    throttle_queue_executor::throttle_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::thread_pool_executor<
            policies::throttle_queue_scheduler<> >(
                max_punits, min_punits))
    {}
#endif

    ///////////////////////////////////////////////////////////////////////////
    local_priority_queue_executor::local_priority_queue_executor()
      : scheduled_executor(new detail::thread_pool_executor<
            policies::local_priority_queue_scheduler<> >(
                get_os_thread_count(), 1, "local_priority_queue_executor"))
    {}

    local_priority_queue_executor::local_priority_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::thread_pool_executor<
            policies::local_priority_queue_scheduler<> >(
                max_punits, min_punits, "local_priority_queue_executor"))
    {}

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    ///////////////////////////////////////////////////////////////////////////
    static_priority_queue_executor::static_priority_queue_executor()
      : scheduled_executor(new detail::thread_pool_executor<
            policies::static_priority_queue_scheduler<> >(
                get_os_thread_count(), 1, "static_priority_queue_executor"))
    {}

    static_priority_queue_executor::static_priority_queue_executor(
            std::size_t max_punits, std::size_t min_punits)
      : scheduled_executor(new detail::thread_pool_executor<
            policies::static_priority_queue_scheduler<> >(
                max_punits, min_punits, "static_priority_queue_executor"))
    {}
#endif
}}}
