//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/hardware/timestamp.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/scheduler_state.hpp>
#include <hpx/threading_base/thread_data.hpp>

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
#include <hpx/thread_pools/detail/scoped_background_timer.hpp>
#endif

#if defined(HPX_HAVE_APEX)
#include <hpx/threading_base/external_timer.hpp>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace hpx { namespace threads { namespace detail {
    ///////////////////////////////////////////////////////////////////////
    inline void write_new_state_log_debug(std::size_t num_thread,
        thread_data* thrd, thread_schedule_state state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "    //-V128
                    << "thread(" << thrd->get_thread_id() << "), "
                    << "description(" << thrd->get_description() << "), "
                    << "new state(" << get_thread_state_name(state) << "), "
                    << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread,
        thread_data* thrd, thread_schedule_state state, char const* info)
    {
        // log this in any case
        LTM_(warning) << "tfunc(" << num_thread << "): "    //-V128
                      << "thread(" << thrd->get_thread_id() << "), "
                      << "description(" << thrd->get_description() << "), "
                      << "new state(" << get_thread_state_name(state) << "), "
                      << info;
    }
    inline void write_old_state_log(
        std::size_t num_thread, thread_data* thrd, thread_schedule_state state)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "    //-V128
                    << "thread(" << thrd->get_thread_id() << "), "
                    << "description(" << thrd->get_description() << "), "
                    << "old state(" << get_thread_state_name(state) << ")";
    }

    ///////////////////////////////////////////////////////////////////////
    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status(thread_data* t, thread_state prev_state)
          : thread_(t)
          , prev_state_(prev_state)
          , next_thread_id_(nullptr)
          , need_restore_state_(t->set_state_tagged(
                thread_schedule_state::active, prev_state_, orig_state_))
        {
        }

        ~switch_status()
        {
            if (need_restore_state_)
            {
                store_state(prev_state_);
            }
        }

        bool is_valid() const
        {
            return need_restore_state_;
        }

        // allow to change the state the thread will be switched to after
        // execution
        thread_state operator=(thread_result_type&& new_state)
        {
            prev_state_ = thread_state(
                new_state.first, prev_state_.state_ex(), prev_state_.tag() + 1);
            next_thread_id_ = std::move(new_state.second);
            return prev_state_;
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
        // HPX-thread in the meantime.
        thread_schedule_state get_previous() const
        {
            return prev_state_.state();
        }

        // This restores the previous state, while making sure that the
        // original state has not been changed since we started executing this
        // thread. The function returns true if the state has been set, false
        // otherwise.
        bool store_state(thread_state& newstate)
        {
            disable_restore();

            if (thread_->restore_state(prev_state_, orig_state_))
            {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore()
        {
            need_restore_state_ = false;
        }

        thread_data* get_next_thread() const
        {
            // we know that the thread-id is just the pointer to the thread_data
            return get_thread_id_data(next_thread_id_);
        }

    private:
        thread_data* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        thread_id_type next_thread_id_;
        bool need_restore_state_;
    };

    class switch_status_background
    {
    public:
        switch_status_background(thread_data* t, thread_state prev_state)
          : thread_(t)
          , prev_state_(prev_state)
          , next_thread_id_(nullptr)
          , need_restore_state_(
                t->set_state_tagged(thread_schedule_state::active, prev_state_,
                    orig_state_, std::memory_order_relaxed))
        {
        }

        ~switch_status_background()
        {
            if (need_restore_state_)
                store_state(prev_state_);
        }

        bool is_valid() const
        {
            return need_restore_state_;
        }

        // allow to change the state the thread will be switched to after
        // execution
        thread_state operator=(thread_result_type&& new_state)
        {
            prev_state_ = thread_state(
                new_state.first, prev_state_.state_ex(), prev_state_.tag() + 1);
            next_thread_id_ = std::move(new_state.second);
            return prev_state_;
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
        // HPX-thread in the meantime.
        thread_schedule_state get_previous() const
        {
            return prev_state_.state();
        }

        // This restores the previous state, while making sure that the
        // original state has not been changed since we started executing this
        // thread. The function returns true if the state has been set, false
        // otherwise.
        bool store_state(thread_state& newstate)
        {
            disable_restore();
            if (thread_->restore_state(prev_state_, orig_state_,
                    std::memory_order_relaxed, std::memory_order_relaxed))
            {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore()
        {
            need_restore_state_ = false;
        }

        thread_data* get_next_thread() const
        {
            // we know that the thread-id is just the pointer to the thread_data
            return get_thread_id_data(next_thread_id_);
        }

    private:
        thread_data* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        thread_id_type next_thread_id_;
        bool need_restore_state_;
    };

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    struct idle_collect_rate
    {
        idle_collect_rate(std::int64_t& tfunc_time, std::int64_t& exec_time)
          : start_timestamp_(util::hardware::timestamp())
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
        {
        }

        void collect_exec_time(std::int64_t timestamp)
        {
            exec_time_ += util::hardware::timestamp() - timestamp;
        }
        void take_snapshot()
        {
            if (tfunc_time_ == std::int64_t(-1))
            {
                start_timestamp_ = util::hardware::timestamp();
                tfunc_time_ = 0;
                exec_time_ = 0;
            }
            else
            {
                tfunc_time_ = util::hardware::timestamp() - start_timestamp_;
            }
        }

        std::int64_t start_timestamp_;

        std::int64_t& tfunc_time_;
        std::int64_t& exec_time_;
    };

    struct exec_time_wrapper
    {
        exec_time_wrapper(idle_collect_rate& idle_rate)
          : timestamp_(util::hardware::timestamp())
          , idle_rate_(idle_rate)
        {
        }
        ~exec_time_wrapper()
        {
            idle_rate_.collect_exec_time(timestamp_);
        }

        std::int64_t timestamp_;
        idle_collect_rate& idle_rate_;
    };

    struct tfunc_time_wrapper
    {
        tfunc_time_wrapper(idle_collect_rate& idle_rate)
          : idle_rate_(idle_rate)
        {
        }
        ~tfunc_time_wrapper()
        {
            idle_rate_.take_snapshot();
        }

        idle_collect_rate& idle_rate_;
    };
#else
    struct idle_collect_rate
    {
        idle_collect_rate(std::int64_t&, std::int64_t&) {}
    };

    struct exec_time_wrapper
    {
        exec_time_wrapper(idle_collect_rate&) {}
    };

    struct tfunc_time_wrapper
    {
        tfunc_time_wrapper(idle_collect_rate&) {}
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct is_active_wrapper
    {
        is_active_wrapper(bool& is_active)
          : is_active_(is_active)
        {
            is_active = true;
        }
        ~is_active_wrapper()
        {
            is_active_ = false;
        }

        bool& is_active_;
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
    struct scheduling_counters
    {
        scheduling_counters(std::int64_t& executed_threads,
            std::int64_t& executed_thread_phases, std::int64_t& tfunc_time,
            std::int64_t& exec_time, std::int64_t& idle_loop_count,
            std::int64_t& busy_loop_count, bool& is_active,
            std::int64_t& background_work_duration,
            std::int64_t& background_send_duration,
            std::int64_t& background_receive_duration)
          : executed_threads_(executed_threads)
          , executed_thread_phases_(executed_thread_phases)
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
          , idle_loop_count_(idle_loop_count)
          , busy_loop_count_(busy_loop_count)
          , background_work_duration_(background_work_duration)
          , background_send_duration_(background_send_duration)
          , background_receive_duration_(background_receive_duration)
          , is_active_(is_active)
        {
        }

        std::int64_t& executed_threads_;
        std::int64_t& executed_thread_phases_;
        std::int64_t& tfunc_time_;
        std::int64_t& exec_time_;
        std::int64_t& idle_loop_count_;
        std::int64_t& busy_loop_count_;
        std::int64_t& background_work_duration_;
        std::int64_t& background_send_duration_;
        std::int64_t& background_receive_duration_;
        bool& is_active_;
    };
#else
    struct scheduling_counters
    {
        scheduling_counters(std::int64_t& executed_threads,
            std::int64_t& executed_thread_phases, std::int64_t& tfunc_time,
            std::int64_t& exec_time, std::int64_t& idle_loop_count,
            std::int64_t& busy_loop_count, bool& is_active)
          : executed_threads_(executed_threads)
          , executed_thread_phases_(executed_thread_phases)
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
          , idle_loop_count_(idle_loop_count)
          , busy_loop_count_(busy_loop_count)
          , is_active_(is_active)
        {
        }

        std::int64_t& executed_threads_;
        std::int64_t& executed_thread_phases_;
        std::int64_t& tfunc_time_;
        std::int64_t& exec_time_;
        std::int64_t& idle_loop_count_;
        std::int64_t& busy_loop_count_;
        bool& is_active_;
    };

#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

    struct scheduling_callbacks
    {
        typedef util::unique_function_nonser<void()> callback_type;
        typedef util::unique_function_nonser<bool()> background_callback_type;

        explicit scheduling_callbacks(callback_type&& outer,
            callback_type&& inner = callback_type(),
            background_callback_type&& background = background_callback_type(),
            std::size_t max_background_threads =
                (std::numeric_limits<std::size_t>::max)(),
            std::size_t max_idle_loop_count = HPX_IDLE_LOOP_COUNT_MAX,
            std::size_t max_busy_loop_count = HPX_BUSY_LOOP_COUNT_MAX)
          : outer_(std::move(outer))
          , inner_(std::move(inner))
          , background_(std::move(background))
          , max_background_threads_(max_background_threads)
          , max_idle_loop_count_(max_idle_loop_count)
          , max_busy_loop_count_(max_busy_loop_count)
        {
        }

        callback_type outer_;
        callback_type inner_;
        background_callback_type background_;
        std::size_t const max_background_threads_;
        std::int64_t const max_idle_loop_count_;
        std::int64_t const max_busy_loop_count_;
    };

    template <typename SchedulingPolicy>
    thread_id_type create_background_thread(SchedulingPolicy& scheduler,
        scheduling_callbacks& callbacks,
        std::shared_ptr<bool>& background_running,
        threads::thread_schedule_hint schedulehint,
        std::int64_t& idle_loop_count)
    {
        thread_id_type background_thread;
        background_running.reset(new bool(true));
        thread_init_data background_init(
            [&, background_running](
                thread_restart_state) -> thread_result_type {
                while (*background_running)
                {
                    if (callbacks.background_())
                    {
                        // we only update the idle_loop_count if
                        // background_running is true. If it was false, this task
                        // was given back to the scheduler.
                        if (*background_running)
                            idle_loop_count = callbacks.max_idle_loop_count_;
                    }
                    // Force yield...
                    hpx::execution_base::this_thread::yield("background_work");
                }

                return thread_result_type(
                    thread_schedule_state::terminated, invalid_thread_id);
            },
            hpx::util::thread_description("background_work"),
            thread_priority::high_recursive, schedulehint,
            thread_stacksize::large,
            // Create in suspended to prevent the thread from being scheduled
            // directly...
            thread_schedule_state::suspended, true, &scheduler);

        scheduler.SchedulingPolicy::create_thread(
            background_init, &background_thread, hpx::throws);
        HPX_ASSERT(background_thread);
        scheduler.SchedulingPolicy::increment_background_thread_count();
        // We can now set the state to pending
        get_thread_id_data(background_thread)
            ->set_state(thread_schedule_state::pending);
        return background_thread;
    }

    // This function tries to invoke the background work thread. It returns
    // false when we need to give the background thread back to scheduler
    // and create a new one that is supposed to be executed inside the
    // scheduling_loop, true otherwise
    template <typename SchedulingPolicy>
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
    bool call_background_thread(thread_id_type& background_thread,
        thread_data*& next_thrd, SchedulingPolicy& scheduler,
        std::size_t num_thread, bool /* running */,
        std::int64_t& background_work_exec_time_init,
        hpx::execution_base::this_thread::detail::agent_storage*
            context_storage)
#else
    bool call_background_thread(thread_id_type& background_thread,
        thread_data*& next_thrd, SchedulingPolicy& scheduler,
        std::size_t num_thread, bool /* running */,
        hpx::execution_base::this_thread::detail::agent_storage*
            context_storage)
#endif
    {
        if (HPX_UNLIKELY(background_thread))
        {
            thread_state state =
                get_thread_id_data(background_thread)->get_state();
            thread_schedule_state state_val = state.state();

            if (HPX_LIKELY(thread_schedule_state::pending == state_val))
            {
                {
                    // tries to set state to active (only if state is still
                    // the same as 'state')
                    detail::switch_status_background thrd_stat(
                        get_thread_id_data(background_thread), state);

                    if (HPX_LIKELY(thrd_stat.is_valid() &&
                            thrd_stat.get_previous() ==
                                thread_schedule_state::pending))
                    {
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
                        // measure background work duration
                        background_work_duration_counter bg_work_duration(
                            background_work_exec_time_init);
                        background_exec_time_wrapper bg_exec_time(
                            bg_work_duration);
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

                        // invoke background thread
                        thrd_stat = (*get_thread_id_data(background_thread))(
                            context_storage);

                        thread_data* next = thrd_stat.get_next_thread();
                        if (next != nullptr &&
                            next != get_thread_id_data(background_thread))
                        {
                            if (next_thrd == nullptr)
                            {
                                next_thrd = next;
                            }
                            else
                            {
                                next->get_scheduler_base()->schedule_thread(
                                    next,
                                    threads::thread_schedule_hint(
                                        static_cast<std::int16_t>(num_thread)),
                                    true);
                                next->get_scheduler_base()->do_some_work(
                                    num_thread);
                            }
                        }
                    }
                    thrd_stat.store_state(state);
                    state_val = state.state();

                    if (HPX_LIKELY(
                            state_val == thread_schedule_state::pending_boost))
                    {
                        get_thread_id_data(background_thread)
                            ->set_state(thread_schedule_state::pending);
                    }
                    else if (thread_schedule_state::terminated == state_val)
                    {
                        scheduler.SchedulingPolicy::
                            decrement_background_thread_count();
                        scheduler.SchedulingPolicy::destroy_thread(
                            get_thread_id_data(background_thread));
                        background_thread.reset();
                    }
                    else if (thread_schedule_state::suspended == state_val)
                    {
                        return false;
                    }
                }
                return true;
            }
            // This should never be reached ... we should only deal with pending
            // here.
            HPX_ASSERT(false);
        }
        return true;
    }

    template <typename SchedulingPolicy>
    void scheduling_loop(std::size_t num_thread, SchedulingPolicy& scheduler,
        scheduling_counters& counters, scheduling_callbacks& params)
    {
        std::atomic<hpx::state>& this_state = scheduler.get_state(num_thread);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        util::itt::stack_context ctx;    // helper for itt support
        util::itt::thread_domain thread_domain;
        util::itt::id threadid(thread_domain, &scheduler);
        util::itt::string_handle task_id("task_id");
        util::itt::string_handle task_phase("task_phase");
        // util::itt::frame_context fctx(thread_domain);
#endif

        std::int64_t& idle_loop_count = counters.idle_loop_count_;
        std::int64_t& busy_loop_count = counters.busy_loop_count_;

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
        std::int64_t& bg_work_exec_time_init =
            counters.background_work_duration_;
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

        idle_collect_rate idle_rate(counters.tfunc_time_, counters.exec_time_);
        tfunc_time_wrapper tfunc_time_collector(idle_rate);

        // spin for some time after queues have become empty
        bool may_exit = false;

#if defined(HPX_HAVE_NETWORKING)
        std::shared_ptr<bool> background_running = nullptr;
        thread_id_type background_thread;

        if (scheduler.SchedulingPolicy::has_scheduler_mode(
                policies::do_background_work) &&
            num_thread < params.max_background_threads_ &&
            !params.background_.empty())
        {
            background_thread =
                create_background_thread(scheduler, params, background_running,
                    thread_schedule_hint(static_cast<std::int16_t>(num_thread)),
                    idle_loop_count);
        }
#endif

        hpx::execution_base::this_thread::detail::agent_storage*
            context_storage =
                hpx::execution_base::this_thread::detail::get_agent_storage();

        std::size_t added = std::size_t(-1);
        thread_data* next_thrd = nullptr;
        while (true)
        {
            thread_data* thrd = next_thrd;
            // Get the next HPX thread from the queue
            bool running =
                this_state.load(std::memory_order_relaxed) < state_pre_sleep;

            // extract the stealing mode once per loop iteration
            bool enable_stealing =
                scheduler.SchedulingPolicy::has_scheduler_mode(
                    policies::enable_stealing);

            // stealing staged threads is enabled if:
            // - fast idle mode is on: same as normal stealing
            // - fast idle mode off: only after normal stealing has failed for
            //                       a while
            bool enable_stealing_staged = enable_stealing;
            if (!scheduler.SchedulingPolicy::has_scheduler_mode(
                    policies::fast_idle_mode))
            {
                enable_stealing_staged = enable_stealing_staged &&
                    idle_loop_count < params.max_idle_loop_count_ / 2;
            }

            if (HPX_LIKELY(thrd ||
                    scheduler.SchedulingPolicy::get_next_thread(
                        num_thread, running, thrd, enable_stealing)))
            {
                tfunc_time_wrapper tfunc_time_collector(idle_rate);
                HPX_ASSERT(thrd->get_scheduler_base() == &scheduler);

                idle_loop_count = params.max_idle_loop_count_;
                ++busy_loop_count;

                may_exit = false;

                // Only pending HPX threads will be executed.
                // Any non-pending HPX threads are leftovers from a set_state()
                // call for a previously pending HPX thread (see comments above).
                thread_state state = thrd->get_state();
                thread_schedule_state state_val = state.state();

                detail::write_old_state_log(num_thread, thrd, state_val);

                if (HPX_LIKELY(thread_schedule_state::pending == state_val))
                {
                    // switch the state of the thread to active and back to
                    // what the thread reports as its return value

                    {
                        // tries to set state to active (only if state is still
                        // the same as 'state')
                        detail::switch_status thrd_stat(thrd, state);
                        if (HPX_LIKELY(thrd_stat.is_valid() &&
                                thrd_stat.get_previous() ==
                                    thread_schedule_state::pending))
                        {
                            tfunc_time_wrapper tfunc_time_collector(idle_rate);

                            // thread returns new required state
                            // store the returned state in the thread
                            {
                                is_active_wrapper utilization(
                                    counters.is_active_);
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
                                util::itt::caller_context cctx(ctx);
                                // util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task =
                                    thrd->get_description().get_task_itt(
                                        thread_domain);
                                task.add_metadata(task_id, thrd);
                                task.add_metadata(
                                    task_phase, thrd->get_thread_phase());
#endif
                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                exec_time_wrapper exec_time_collector(
                                    idle_rate);

#if defined(HPX_HAVE_APEX)
                                // get the APEX data pointer, in case we are resuming the
                                // thread and have to restore any leaf timers from
                                // direct actions, etc.

                                // the address of tmp_data is getting stored
                                // by APEX during this call
                                util::external_timer::scoped_timer profiler(
                                    thrd->get_timer_data());

                                thrd_stat = (*thrd)(context_storage);

                                if (thrd_stat.get_previous() ==
                                    thread_schedule_state::terminated)
                                {
                                    profiler.stop();
                                    // just in case, clean up the now dead pointer.
                                    thrd->set_timer_data(nullptr);
                                }
                                else
                                {
                                    profiler.yield();
                                }
#else
                                thrd_stat = (*thrd)(context_storage);
#endif
                            }

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                            ++counters.executed_thread_phases_;
#endif
                        }
                        else
                        {
                            // some other worker-thread got in between and started
                            // executing this HPX-thread, we just continue with
                            // the next one
                            thrd_stat.disable_restore();
                            detail::write_new_state_log_warning(
                                num_thread, thrd, state_val, "no execution");
                            continue;
                        }

                        // store and retrieve the new state in the thread
                        if (HPX_UNLIKELY(!thrd_stat.store_state(state)))
                        {
                            // some other worker-thread got in between and changed
                            // the state of this thread, we just continue with
                            // the next one
                            detail::write_new_state_log_warning(
                                num_thread, thrd, state_val, "no state change");
                            continue;
                        }

                        state_val = state.state();

                        // any exception thrown from the thread will reset its
                        // state at this point

                        // handle next thread id if given (switch directly to
                        // this thread)
                        next_thrd = thrd_stat.get_next_thread();
                    }

                    //detail::write_new_state_log_debug(num_thread, thrd,
                    //    state_val, "normal");

                    // Re-add this work item to our list of work items if the HPX
                    // thread should be re-scheduled. If the HPX thread is suspended
                    // now we just keep it in the map of threads.
                    if (HPX_UNLIKELY(
                            state_val == thread_schedule_state::pending))
                    {
                        if (HPX_LIKELY(next_thrd == nullptr))
                        {
                            // schedule other work
                            scheduler.SchedulingPolicy::wait_or_add_new(
                                num_thread, running, idle_loop_count,
                                enable_stealing_staged, added);
                        }

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        scheduler.SchedulingPolicy::schedule_thread_last(thrd,
                            threads::thread_schedule_hint(
                                static_cast<std::int16_t>(num_thread)),
                            true);
                        scheduler.SchedulingPolicy::do_some_work(num_thread);
                    }
                    else if (HPX_UNLIKELY(state_val ==
                                 thread_schedule_state::pending_boost))
                    {
                        thrd->set_state(thread_schedule_state::pending);

                        if (HPX_LIKELY(next_thrd == nullptr))
                        {
                            // reschedule this thread right away if the
                            // background work will be triggered
                            if (HPX_UNLIKELY(busy_loop_count >
                                    params.max_busy_loop_count_))
                            {
                                next_thrd = thrd;
                            }
                            else
                            {
                                // schedule other work
                                scheduler.SchedulingPolicy::wait_or_add_new(
                                    num_thread, running, idle_loop_count,
                                    enable_stealing_staged, added);

                                // schedule this thread again immediately with
                                // boosted priority
                                scheduler.SchedulingPolicy::schedule_thread(
                                    thrd,
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
                            scheduler.SchedulingPolicy::schedule_thread(thrd,
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
                    LTM_(warning) << "tfunc(" << num_thread
                                  << "): "    //-V128
                                     "thread("
                                  << thrd->get_thread_id()
                                  << "), "
                                     "description("
                                  << thrd->get_description()
                                  << "), "
                                     "rescheduling";

                    // re-schedule thread, if it is still marked as active
                    // this might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset
                    // yet
                    scheduler.SchedulingPolicy::schedule_thread(thrd,
                        threads::thread_schedule_hint(
                            static_cast<std::int16_t>(num_thread)),
                        true, thrd->get_priority());
                    scheduler.SchedulingPolicy::do_some_work(num_thread);
                }

                // Remove the mapping from thread_map_ if HPX thread is depleted
                // or terminated, this will delete the HPX thread.
                // REVIEW: what has to be done with depleted HPX threads?
                if (HPX_LIKELY(state_val == thread_schedule_state::depleted ||
                        state_val == thread_schedule_state::terminated))
                {
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                    ++counters.executed_threads_;
#endif
                    scheduler.SchedulingPolicy::destroy_thread(thrd);
                }
            }

            // if nothing else has to be done either wait or terminate
            else
            {
                --idle_loop_count;

                if (scheduler.SchedulingPolicy::wait_or_add_new(num_thread,
                        running, idle_loop_count, enable_stealing_staged,
                        added))
                {
                    // Clean up terminated threads before trying to exit
                    bool can_exit = !running &&
                        scheduler.SchedulingPolicy::cleanup_terminated(
                            num_thread, true) &&
                        scheduler.SchedulingPolicy::get_queue_length(
                            num_thread) == 0;

                    if (this_state.load() == state_pre_sleep)
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
                            if (!scheduler.SchedulingPolicy::has_scheduler_mode(
                                    policies::delay_exit))
                            {
                                // If this is an inner scheduler, try to exit immediately
#if defined(HPX_HAVE_NETWORKING)
                                if (background_thread != nullptr)
                                {
                                    HPX_ASSERT(background_running);
                                    *background_running = false;
                                    scheduler.SchedulingPolicy::
                                        decrement_background_thread_count();
                                    scheduler.SchedulingPolicy::schedule_thread(
                                        get_thread_id_data(background_thread),
                                        threads::thread_schedule_hint(
                                            static_cast<std::int16_t>(
                                                num_thread)),
                                        true,
                                        get_thread_id_data(background_thread)
                                            ->get_priority());
                                    scheduler.SchedulingPolicy::do_some_work(
                                        num_thread);

                                    background_thread.reset();
                                    background_running.reset();
                                }
                                else
#endif
                                {
                                    this_state.store(state_stopped);
                                    break;
                                }
                            }
                            else
                            {
                                // Otherwise, keep idling for some time
                                if (!may_exit)
                                    idle_loop_count =
                                        params.max_idle_loop_count_;
                                may_exit = true;
                            }
                        }
                    }
                }
                else if (!may_exit && added == 0 &&
                    (scheduler.SchedulingPolicy::has_scheduler_mode(
                        policies::fast_idle_mode)))
                {
                    // speed up idle suspend if no work was stolen
                    idle_loop_count -= params.max_idle_loop_count_ / 256;
                    added = std::size_t(-1);
                }

#if defined(HPX_HAVE_NETWORKING)
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
                // do background work in parcel layer and in agas
                if (!call_background_thread(background_thread, next_thrd,
                        scheduler, num_thread, running, bg_work_exec_time_init,
                        context_storage))
#else
                if (!call_background_thread(background_thread, next_thrd,
                        scheduler, num_thread, running, context_storage))
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS
                {
                    // Let the current background thread terminate as soon as
                    // possible. No need to reschedule, as another LCO will
                    // set it to pending and schedule it back eventually
                    HPX_ASSERT(background_thread);
                    HPX_ASSERT(background_running);
                    *background_running = false;
                    scheduler
                        .SchedulingPolicy::decrement_background_thread_count();
                    // Create a new one which will replace the current such we
                    // avoid deadlock situations, if all background threads are
                    // blocked.
                    background_thread = create_background_thread(scheduler,
                        params, background_running,
                        thread_schedule_hint(
                            static_cast<std::int16_t>(num_thread)),
                        idle_loop_count);
                }
#endif
                // call back into invoking context
                if (!params.inner_.empty())
                {
                    params.inner_();
                    context_storage = hpx::execution_base::this_thread::detail::
                        get_agent_storage();
                }
            }

            scheduler.custom_polling_function();

            // something went badly wrong, give up
            if (HPX_UNLIKELY(this_state.load() == state_terminating))
                break;

            if (busy_loop_count > params.max_busy_loop_count_)
            {
                busy_loop_count = 0;

#if defined(HPX_HAVE_NETWORKING)
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) &&                            \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
                // do background work in parcel layer and in agas
                if (!call_background_thread(background_thread, next_thrd,
                        scheduler, num_thread, running, bg_work_exec_time_init,
                        context_storage))
#else
                // do background work in parcel layer and in agas
                if (!call_background_thread(background_thread, next_thrd,
                        scheduler, num_thread, running, context_storage))
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS
                {
                    // Let the current background thread terminate as soon
                    // as possible. No need to reschedule, as another LCO
                    // will set it to pending and schedule it back eventually
                    HPX_ASSERT(background_thread);
                    HPX_ASSERT(background_running);
                    *background_running = false;
                    scheduler
                        .SchedulingPolicy::decrement_background_thread_count();
                    // Create a new one which will replace the current such
                    // we avoid deadlock situations, if all background
                    // threads are blocked.
                    background_thread = create_background_thread(scheduler,
                        params, background_running,
                        thread_schedule_hint(
                            static_cast<std::int16_t>(num_thread)),
                        idle_loop_count);
                }
#endif
            }
            else if (idle_loop_count < 0 || may_exit)
            {
                if (idle_loop_count < 0)
                    idle_loop_count = params.max_idle_loop_count_;

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
                    HPX_ASSERT(this_state.load() != state_pre_sleep);

#if defined(HPX_HAVE_NETWORKING)
                    if (background_thread)
                    {
                        HPX_ASSERT(background_running);
                        *background_running = false;
                        scheduler.SchedulingPolicy::
                            decrement_background_thread_count();
                        scheduler.SchedulingPolicy::schedule_thread(
                            get_thread_id_data(background_thread),
                            threads::thread_schedule_hint(
                                static_cast<std::int16_t>(num_thread)),
                            true,
                            get_thread_id_data(background_thread)
                                ->get_priority());
                        scheduler.SchedulingPolicy::do_some_work(num_thread);
                        background_thread.reset();
                        background_running.reset();
                    }
                    else
#endif
                    {
                        bool can_exit = !running &&
                            scheduler.SchedulingPolicy::cleanup_terminated(
                                true) &&
                            scheduler.SchedulingPolicy::get_thread_count(
                                thread_schedule_state::suspended,
                                thread_priority::default_, num_thread) == 0 &&
                            scheduler.SchedulingPolicy::get_queue_length(
                                num_thread) == 0;

                        if (can_exit)
                        {
                            this_state.store(state_stopped);
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
}}}    // namespace hpx::threads::detail

// NOTE: This line only exists to please doxygen. Without the line doxygen
// generates incomplete xml output.
