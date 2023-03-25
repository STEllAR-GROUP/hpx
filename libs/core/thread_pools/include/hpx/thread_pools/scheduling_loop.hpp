//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/thread_pools/detail/background_thread.hpp>
#include <hpx/thread_pools/detail/scheduling_callbacks.hpp>
#include <hpx/thread_pools/detail/scheduling_counters.hpp>
#include <hpx/thread_pools/detail/scheduling_log.hpp>
#include <hpx/threading_base/detail/switch_status.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/scheduler_state.hpp>
#include <hpx/threading_base/thread_data.hpp>

#if defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx::threads::detail {

    ///////////////////////////////////////////////////////////////////////
#ifdef HPX_HAVE_THREAD_IDLE_RATES
    struct idle_collect_rate
    {
        idle_collect_rate(
            std::int64_t& tfunc_time, std::int64_t& exec_time) noexcept
          : start_timestamp_(util::hardware::timestamp())
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
        {
        }

        void collect_exec_time(std::uint64_t timestamp) const noexcept
        {
            exec_time_ += static_cast<std::int64_t>(
                util::hardware::timestamp() - timestamp);
        }

        void take_snapshot() noexcept
        {
            if (tfunc_time_ == static_cast<std::int64_t>(-1))
            {
                start_timestamp_ = util::hardware::timestamp();
                tfunc_time_ = 0;
                exec_time_ = 0;
            }
            else
            {
                tfunc_time_ = static_cast<std::int64_t>(
                    util::hardware::timestamp() - start_timestamp_);
            }
        }

        std::uint64_t start_timestamp_;

        std::int64_t& tfunc_time_;
        std::int64_t& exec_time_;
    };

    struct exec_time_wrapper
    {
        explicit exec_time_wrapper(idle_collect_rate& idle_rate) noexcept
          : timestamp_(util::hardware::timestamp())
          , idle_rate_(idle_rate)
        {
        }

        exec_time_wrapper(exec_time_wrapper const&) = delete;
        exec_time_wrapper(exec_time_wrapper&&) = delete;

        exec_time_wrapper& operator=(exec_time_wrapper const&) = delete;
        exec_time_wrapper& operator=(exec_time_wrapper&&) = delete;

        ~exec_time_wrapper()
        {
            idle_rate_.collect_exec_time(timestamp_);
        }

        std::uint64_t timestamp_;
        idle_collect_rate& idle_rate_;
    };

    struct tfunc_time_wrapper
    {
        explicit constexpr tfunc_time_wrapper(
            idle_collect_rate& idle_rate) noexcept
          : idle_rate_(idle_rate)
        {
        }

        tfunc_time_wrapper(tfunc_time_wrapper const&) = delete;
        tfunc_time_wrapper(tfunc_time_wrapper&&) = delete;

        tfunc_time_wrapper& operator=(tfunc_time_wrapper const&) = delete;
        tfunc_time_wrapper& operator=(tfunc_time_wrapper&&) = delete;

        ~tfunc_time_wrapper()
        {
            idle_rate_.take_snapshot();
        }

        idle_collect_rate& idle_rate_;
    };
#else
    struct idle_collect_rate
    {
        explicit constexpr idle_collect_rate(
            std::int64_t&, std::int64_t&) noexcept
        {
        }
    };

    struct exec_time_wrapper
    {
        explicit constexpr exec_time_wrapper(idle_collect_rate&) noexcept {}
    };

    struct tfunc_time_wrapper
    {
        explicit constexpr tfunc_time_wrapper(idle_collect_rate&) noexcept {}
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct is_active_wrapper
    {
        explicit is_active_wrapper(bool& is_active) noexcept
          : is_active_(is_active)
        {
            is_active = true;
        }

        is_active_wrapper(is_active_wrapper const&) = delete;
        is_active_wrapper(is_active_wrapper&&) = delete;

        is_active_wrapper& operator=(is_active_wrapper const&) = delete;
        is_active_wrapper& operator=(is_active_wrapper&&) = delete;

        ~is_active_wrapper()
        {
            is_active_ = false;
        }

        bool& is_active_;
    };

    template <typename SchedulingPolicy>
    void scheduling_loop(std::size_t num_thread, SchedulingPolicy& scheduler,
        scheduling_counters& counters, scheduling_callbacks& params)
    {
        std::atomic<hpx::state>& this_state = scheduler.get_state(num_thread);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        util::itt::stack_context ctx;    // helper for itt support
        util::itt::thread_domain const thread_domain;
        util::itt::id threadid(thread_domain, &scheduler);
        util::itt::string_handle const task_id("task_id");
        util::itt::string_handle const task_phase("task_phase");
        // util::itt::frame_context fctx(thread_domain);
#endif

        std::int64_t& idle_loop_count = counters.idle_loop_count_;
        std::int64_t& busy_loop_count = counters.busy_loop_count_;

        background_work_exec_time bg_work_exec_time_init(counters);

        idle_collect_rate idle_rate(counters.tfunc_time_, counters.exec_time_);
        [[maybe_unused]] tfunc_time_wrapper tfunc_time_collector(idle_rate);

        // spin for some time after queues have become empty
        bool may_exit = false;

        std::shared_ptr<bool> background_running;
        thread_id_ref_type background_thread;

        if (scheduler.has_scheduler_mode(
                policies::scheduler_mode::do_background_work) &&
            num_thread < params.max_background_threads_ &&
            !params.background_.empty())
        {
            background_thread = create_background_thread(scheduler, num_thread,
                params, background_running, idle_loop_count);
        }

        hpx::execution_base::this_thread::detail::agent_storage*
            context_storage =
                hpx::execution_base::this_thread::detail::get_agent_storage();

        auto added = static_cast<std::size_t>(-1);
        thread_id_ref_type next_thrd;
        while (true)
        {
            thread_id_ref_type thrd = HPX_MOVE(next_thrd);
            next_thrd = thread_id_ref_type();

            // Get the next HPX thread from the queue
            bool running = this_state.load(std::memory_order_relaxed) <
                hpx::state::pre_sleep;

            // extract the stealing mode once per loop iteration (except during
            // shutdown)
            bool enable_stealing = !may_exit &&
                scheduler.has_scheduler_mode(
                    policies::scheduler_mode::enable_stealing);

            // stealing staged threads is enabled if:
            // - fast idle mode is on: same as normal stealing
            // - fast idle mode off: only after normal stealing has failed for
            //                       a while
            bool enable_stealing_staged = enable_stealing;
            if (enable_stealing_staged &&
                !scheduler.has_scheduler_mode(
                    policies::scheduler_mode::fast_idle_mode))
            {
                enable_stealing_staged = !may_exit &&
                    idle_loop_count > params.max_idle_loop_count_ / 2;
            }

            if (HPX_LIKELY(thrd ||
                    scheduler.get_next_thread(
                        num_thread, running, thrd, enable_stealing)))
            {
                HPX_ASSERT(get_thread_id_data(thrd)->get_scheduler_base() ==
                    &scheduler);

                idle_loop_count = 0;
                ++busy_loop_count;

                may_exit = false;

                // Only pending HPX threads will be executed. Any non-pending
                // HPX threads are leftovers from a set_state() call for a
                // previously pending HPX thread (see comments above).
                auto* thrdptr = get_thread_id_data(thrd);
                thread_state state = thrdptr->get_state();
                thread_schedule_state state_val = state.state();

                if (HPX_LIKELY(thread_schedule_state::pending == state_val))
                {
                    // switch the state of the thread to active and back to what
                    // the thread reports as its return value

                    {
                        // tries to set state to active (only if state is still
                        // the same as 'state')
                        detail::switch_status thrd_stat(thrd, state);
                        if (HPX_LIKELY(thrd_stat.is_valid() &&
                                thrd_stat.get_previous() ==
                                    thread_schedule_state::pending))
                        {
                            detail::write_state_log(scheduler, num_thread, thrd,
                                thrd_stat.get_previous(),
                                thread_schedule_state::active);

                            [[maybe_unused]] tfunc_time_wrapper
                                tfunc_time_collector_inner(idle_rate);

                            // thread returns new required state store the
                            // returned state in the thread
                            {
                                is_active_wrapper utilization(
                                    counters.is_active_);
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
                                util::itt::caller_context cctx(ctx);
                                // util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task =
                                    thrdptr->get_description().get_task_itt(
                                        thread_domain);
                                task.add_metadata(task_id, thrdptr);
                                task.add_metadata(
                                    task_phase, thrdptr->get_thread_phase());
#endif
                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                [[maybe_unused]] exec_time_wrapper
                                    exec_time_collector(idle_rate);

#if defined(HPX_HAVE_APEX)
                                // get the APEX data pointer, in case we are
                                // resuming the thread and have to restore any
                                // leaf timers from direct actions, etc.

                                // the address of tmp_data is getting stored by
                                // APEX during this call
                                util::external_timer::scoped_timer profiler(
                                    thrdptr->get_timer_data());

                                thrd_stat = (*thrdptr)(context_storage);

                                if (thrd_stat.get_previous() ==
                                    thread_schedule_state::terminated)
                                {
                                    profiler.stop();
                                    // just in case, clean up the now dead pointer.
                                    thrdptr->set_timer_data(nullptr);
                                }
                                else
                                {
                                    profiler.yield();
                                }
#else
                                thrd_stat = (*thrdptr)(context_storage);
#endif
                            }

                            detail::write_state_log(scheduler, num_thread, thrd,
                                thread_schedule_state::active,
                                thrd_stat.get_previous());

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                            ++counters.executed_thread_phases_;
#endif
                        }
                        else
                        {
                            // some other worker-thread got in between and
                            // started executing this HPX-thread, we just
                            // continue with the next one
                            thrd_stat.disable_restore();
                            detail::write_state_log_warning(scheduler,
                                num_thread, thrd, state_val, "no execution");
                            continue;
                        }

                        // store and retrieve the new state in the thread
                        if (HPX_UNLIKELY(!thrd_stat.store_state(state)))
                        {
                            // some other worker-thread got in between and
                            // changed the state of this thread, we just
                            // continue with the next one
                            detail::write_state_log_warning(scheduler,
                                num_thread, thrd, state_val, "no state change");
                            continue;
                        }

                        state_val = state.state();

                        // any exception thrown from the thread will reset its
                        // state at this point

                        // handle next thread id if given (switch directly to
                        // this thread)
                        next_thrd = thrd_stat.move_next_thread();
                    }

                    // Re-add this work item to our list of work items if the
                    // HPX thread should be re-scheduled. If the HPX thread is
                    // suspended now we just keep it in the map of threads.
                    if (HPX_UNLIKELY(
                            state_val == thread_schedule_state::pending))
                    {
                        if (HPX_LIKELY(next_thrd == nullptr))
                        {
                            // schedule other work
                            scheduler.wait_or_add_new(num_thread, running,
                                idle_loop_count, enable_stealing_staged, added);
                        }

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        scheduler.SchedulingPolicy::schedule_thread_last(
                            HPX_MOVE(thrd),
                            threads::thread_schedule_hint(
                                static_cast<std::int16_t>(num_thread)),
                            true);
                        scheduler.SchedulingPolicy::do_some_work(num_thread);
                    }
                    else if (HPX_UNLIKELY(state_val ==
                                 thread_schedule_state::pending_boost))
                    {
                        thrdptr->set_state(thread_schedule_state::pending);

                        if (HPX_LIKELY(next_thrd == nullptr))
                        {
                            // reschedule this thread right away if the
                            // background work will be triggered
                            if (HPX_UNLIKELY(busy_loop_count >
                                    params.max_busy_loop_count_))
                            {
                                next_thrd = HPX_MOVE(thrd);
                            }
                            else
                            {
                                // schedule other work
                                scheduler.wait_or_add_new(num_thread, running,
                                    idle_loop_count, enable_stealing_staged,
                                    added);

                                // schedule this thread again immediately with
                                // boosted priority
                                scheduler.SchedulingPolicy::schedule_thread(
                                    HPX_MOVE(thrd),
                                    threads::thread_schedule_hint(
                                        static_cast<std::int16_t>(num_thread)),
                                    true, thread_priority::boost);
                                scheduler.SchedulingPolicy::do_some_work(
                                    num_thread);
                            }
                        }
                        else if (HPX_LIKELY(next_thrd != thrd))
                        {
                            // schedule this thread again immediately with
                            // boosted priority
                            scheduler.SchedulingPolicy::schedule_thread(
                                HPX_MOVE(thrd),
                                threads::thread_schedule_hint(
                                    static_cast<std::int16_t>(num_thread)),
                                true, thread_priority::boost);
                            scheduler.SchedulingPolicy::do_some_work(
                                num_thread);
                        }
                    }
                }
                else if (HPX_UNLIKELY(
                             thread_schedule_state::active == state_val))
                {
                    write_rescheduling_log_warning(scheduler, num_thread, thrd);

                    // re-schedule thread, if it is still marked as active this
                    // might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset
                    // yet
                    auto priority = thrdptr->get_priority();
                    scheduler.SchedulingPolicy::schedule_thread(HPX_MOVE(thrd),
                        threads::thread_schedule_hint(
                            static_cast<std::int16_t>(num_thread)),
                        true, priority);
                    scheduler.SchedulingPolicy::do_some_work(num_thread);
                }

                // Remove the mapping from thread_map_ if HPX thread is depleted
                // or terminated, this will delete the HPX thread. REVIEW: what
                // has to be done with depleted HPX threads?
                if (HPX_LIKELY(state_val == thread_schedule_state::depleted ||
                        state_val == thread_schedule_state::terminated))
                {
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                    ++counters.executed_threads_;
#endif
                    thrd = thread_id_type();
                }
            }

            // if nothing else has to be done either wait or terminate
            else
            {
                ++idle_loop_count;

                if (scheduler.wait_or_add_new(num_thread, running,
                        idle_loop_count, enable_stealing_staged, added,
                        &next_thrd))
                {
                    // Clean up terminated threads before trying to exit
                    bool can_exit = !running &&
                        scheduler.SchedulingPolicy::cleanup_terminated(
                            num_thread, true) &&
                        scheduler.SchedulingPolicy::get_queue_length(
                            num_thread) == 0;

                    if (this_state.load(std::memory_order_relaxed) ==
                        hpx::state::pre_sleep)
                    {
                        if (can_exit)
                        {
                            scheduler.SchedulingPolicy::suspend(num_thread);
                        }
                    }
                    else
                    {
                        can_exit = can_exit &&
                            scheduler.SchedulingPolicy::get_thread_count(
                                thread_schedule_state::suspended,
                                thread_priority::default_, num_thread) == 0;

                        if (can_exit)
                        {
                            if (!scheduler.has_scheduler_mode(
                                    policies::scheduler_mode::delay_exit))
                            {
                                // If this is an inner scheduler, try to exit
                                // immediately
                                if (background_thread != nullptr)
                                {
                                    HPX_ASSERT(background_running);
                                    *background_running = false;    //-V522

                                    // do background work in parcel layer and in agas
                                    [[maybe_unused]] bool const has_exited =
                                        call_background_thread(
                                            background_thread, next_thrd,
                                            scheduler, num_thread,
                                            bg_work_exec_time_init,
                                            context_storage);

                                    // the background thread should have exited
                                    HPX_ASSERT(has_exited);

                                    background_thread = thread_id_type();
                                    background_running.reset();
                                }
                                else
                                {
                                    this_state.store(hpx::state::stopped);
                                    break;
                                }
                            }
                            else
                            {
                                // Otherwise, keep idling for some time
                                if (!may_exit)
                                    idle_loop_count = 0;
                                may_exit = true;
                            }
                        }
                    }
                }
                else if (!may_exit && added == 0 &&
                    (scheduler.has_scheduler_mode(
                        policies::scheduler_mode::fast_idle_mode)))
                {
                    // speed up idle suspend if no work was stolen
                    idle_loop_count += params.max_idle_loop_count_ / 1024;
                    added = static_cast<std::size_t>(-1);
                }

                // if stealing yielded a new task, run it first
                if (next_thrd != nullptr)
                {
                    continue;
                }

                // do background work in parcel layer and in agas
                call_and_create_background_thread(background_thread, next_thrd,
                    scheduler, num_thread, bg_work_exec_time_init,
                    context_storage, params, background_running,
                    idle_loop_count);

                // call back into invoking context
                if (!params.inner_.empty())
                {
                    params.inner_();
                    context_storage = hpx::execution_base::this_thread::detail::
                        get_agent_storage();
                }
            }

            if (scheduler.custom_polling_function() ==
                policies::detail::polling_status::busy)
            {
                idle_loop_count = 0;
            }

            // something went badly wrong, give up
            if (HPX_UNLIKELY(this_state.load(std::memory_order_relaxed) ==
                    hpx::state::terminating))
            {
                break;
            }

            if (busy_loop_count > params.max_busy_loop_count_)
            {
                busy_loop_count = 0;

                // do background work in parcel layer and in agas
                call_and_create_background_thread(background_thread, next_thrd,
                    scheduler, num_thread, bg_work_exec_time_init,
                    context_storage, params, background_running,
                    idle_loop_count);
            }
            else if (idle_loop_count > params.max_idle_loop_count_ || may_exit)
            {
                if (idle_loop_count > params.max_idle_loop_count_)
                    idle_loop_count = 0;

                // call back into invoking context
                if (!params.outer_.empty())
                {
                    params.outer_();
                    context_storage = hpx::execution_base::this_thread::detail::
                        get_agent_storage();
                }

                // break if we were idling after 'may_exit'
                if (may_exit)
                {
                    HPX_ASSERT(this_state.load(std::memory_order_relaxed) !=
                        hpx::state::pre_sleep);

                    if (background_thread)
                    {
                        HPX_ASSERT(background_running);
                        *background_running = false;

                        // do background work in parcel layer and in agas
                        [[maybe_unused]] bool const has_exited =
                            call_background_thread(background_thread, next_thrd,
                                scheduler, num_thread, bg_work_exec_time_init,
                                context_storage);

                        // the background thread should have exited
                        HPX_ASSERT(has_exited);

                        background_thread = thread_id_type();
                        background_running.reset();
                    }
                    else
                    {
                        bool const can_exit = !running &&
                            scheduler.SchedulingPolicy::cleanup_terminated(
                                true) &&
                            scheduler.SchedulingPolicy::get_thread_count(
                                thread_schedule_state::suspended,
                                thread_priority::default_, num_thread) == 0 &&
                            scheduler.SchedulingPolicy::get_queue_length(
                                num_thread) == 0;

                        if (can_exit)
                        {
                            this_state.store(hpx::state::stopped);
                            break;
                        }
                    }

                    may_exit = false;
                }
                else
                {
                    scheduler.SchedulingPolicy::cleanup_terminated(true);
                }
            }
        }
    }
}    // namespace hpx::threads::detail

// NOTE: This line only exists to please doxygen. Without the line doxygen
// generates incomplete xml output.
