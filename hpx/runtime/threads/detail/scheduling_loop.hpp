//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM)
#define HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM

#include <hpx/config.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/runtime/threads/detail/periodic_maintenance.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/state.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <boost/atomic.hpp>

#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    inline void write_new_state_log_debug(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): " //-V128
            << "thread(" << thrd->get_thread_id().get() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state, char const* info)
    {
        // log this in any case
        LTM_(warning) << "tfunc(" << num_thread << "): " //-V128
            << "thread(" << thrd->get_thread_id().get() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_old_state_log(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): " //-V128
                    << "thread(" << thrd->get_thread_id().get() << "), "
                    << "description(" << thrd->get_description() << "), "
                    << "old state(" << get_thread_state_name(state) << ")";
    }

    ///////////////////////////////////////////////////////////////////////
    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status (thread_data* t, thread_state prev_state)
          : thread_(t), prev_state_(prev_state),
            next_thread_id_(nullptr),
            need_restore_state_(t->set_state_tagged(active, prev_state_, orig_state_))
        {}

        ~switch_status ()
        {
            if (need_restore_state_)
                store_state(prev_state_);
        }

        bool is_valid() const { return need_restore_state_; }

        // allow to change the state the thread will be switched to after
        // execution
        thread_state operator=(thread_result_type && new_state)
        {
            prev_state_ = thread_state(new_state.first,
                prev_state_.state_ex(), prev_state_.tag() + 1);
            next_thread_id_ = std::move(new_state.second);
            return prev_state_;
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
        // HPX-thread in the meantime.
        thread_state_enum get_previous() const
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
            if (thread_->restore_state(prev_state_, orig_state_)) {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore() { need_restore_state_ = false; }

        thread_data* get_next_thread() const
        {
            // we know that the thread-id is just the pointer to the thread_data
            return reinterpret_cast<thread_data*>(next_thread_id_.get());
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
        idle_collect_rate(std::uint64_t& tfunc_time, std::uint64_t& exec_time)
          : start_timestamp_(util::hardware::timestamp())
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
        {}

        void collect_exec_time(std::uint64_t timestamp)
        {
            exec_time_ += util::hardware::timestamp() - timestamp;
        }
        void take_snapshot()
        {
            if (tfunc_time_ == std::uint64_t(-1))
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

        std::uint64_t start_timestamp_;

        std::uint64_t& tfunc_time_;
        std::uint64_t& exec_time_;
    };

    struct exec_time_wrapper
    {
        exec_time_wrapper(idle_collect_rate& idle_rate)
          : timestamp_(util::hardware::timestamp())
          , idle_rate_(idle_rate)
        {}
        ~exec_time_wrapper()
        {
            idle_rate_.collect_exec_time(timestamp_);
        }

        std::uint64_t timestamp_;
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
        idle_collect_rate(std::uint64_t&, std::uint64_t&) {}
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
        is_active_wrapper(std::uint8_t& is_active)
          : is_active_(is_active)
        {
            is_active = 1;
        }
        ~is_active_wrapper()
        {
            is_active_ = 0;
        }

        std::uint8_t& is_active_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct scheduling_counters
    {
        scheduling_counters(std::int64_t& executed_threads,
                std::int64_t& executed_thread_phases,
                std::uint64_t& tfunc_time, std::uint64_t& exec_time,
                std::int64_t& idle_loop_count, std::int64_t& busy_loop_count,
                std::uint8_t& is_active)
          : executed_threads_(executed_threads),
            executed_thread_phases_(executed_thread_phases),
            tfunc_time_(tfunc_time),
            exec_time_(exec_time),
            idle_loop_count_(idle_loop_count),
            busy_loop_count_(busy_loop_count),
            is_active_(is_active)
        {}

        std::int64_t& executed_threads_;
        std::int64_t& executed_thread_phases_;
        std::uint64_t& tfunc_time_;
        std::uint64_t& exec_time_;
        std::int64_t& idle_loop_count_;
        std::int64_t& busy_loop_count_;
        std::uint8_t& is_active_;
    };

    struct scheduling_callbacks
    {
        typedef util::function_nonser<void()> callback_type;
        typedef util::function_nonser<bool()> background_callback_type;

        explicit scheduling_callbacks(
                callback_type && outer,
                callback_type && inner = callback_type(),
                background_callback_type && background =
                    background_callback_type(),
                std::size_t max_background_threads =
                    hpx::util::safe_lexical_cast<std::size_t>(
                        hpx::get_config_entry("hpx.max_background_threads",
                            (std::numeric_limits<std::size_t>::max)())),
                std::size_t max_idle_loop_count =
                    hpx::util::safe_lexical_cast<std::int64_t>(
                        hpx::get_config_entry("hpx.max_idle_loop_count",
                            HPX_IDLE_LOOP_COUNT_MAX)),
                std::size_t max_busy_loop_count =
                    hpx::util::safe_lexical_cast<std::int64_t>(
                        hpx::get_config_entry("hpx.max_busy_loop_count",
                            HPX_BUSY_LOOP_COUNT_MAX)))
          : outer_(std::move(outer)),
            inner_(std::move(inner)),
            background_(std::move(background)),
            max_background_threads_(max_background_threads),
            max_idle_loop_count_(max_idle_loop_count),
            max_busy_loop_count_(max_busy_loop_count)
        {}

        callback_type outer_;
        callback_type inner_;
        background_callback_type background_;
        std::size_t const max_background_threads_;
        std::int64_t const max_idle_loop_count_;
        std::int64_t const max_busy_loop_count_;
    };

    template <typename SchedulingPolicy>
    void scheduling_loop(std::size_t num_thread, SchedulingPolicy& scheduler,
        scheduling_counters& counters, scheduling_callbacks& params)
    {
        boost::atomic<hpx::state>& this_state = scheduler.get_state(num_thread);

        util::itt::stack_context ctx;        // helper for itt support
        util::itt::domain domain = hpx::get_thread_itt_domain();
        util::itt::id threadid(domain, &scheduler);
        util::itt::string_handle task_id("task_id");
        util::itt::string_handle task_phase("task_phase");

//         util::itt::frame_context fctx(domain);

        std::int64_t& idle_loop_count = counters.idle_loop_count_;
        std::int64_t& busy_loop_count = counters.busy_loop_count_;

        idle_collect_rate idle_rate(counters.tfunc_time_, counters.exec_time_);
        tfunc_time_wrapper tfunc_time_collector(idle_rate);

        scheduler.SchedulingPolicy::start_periodic_maintenance(this_state);

        // spin for some time after queues have become empty
        bool may_exit = false;
        thread_data* thrd = nullptr;
        thread_data* next_thrd = nullptr;

        thread_init_data background_init(
            [&](thread_state_ex_enum) -> thread_result_type
            {
                while(true)
                {
                    if (params.background_())
                        idle_loop_count = 0;
                    hpx::this_thread::suspend(hpx::threads::pending,
                        "background_work");
                }
                return std::make_pair(terminated, nullptr);
            },
            hpx::util::thread_description("background_work"));

        thread_id_type background_thread = nullptr;
        if ((scheduler.get_scheduler_mode() & policies::do_background_work) &&
            num_thread < params.max_background_threads_ &&
            !params.background_.empty())
        {
            background_thread.reset(
                new thread_data(background_init, nullptr, pending));
        }

        while (true) {
            // Get the next HPX thread from the queue
            thrd = next_thrd;
            bool running = this_state.load(
                boost::memory_order_relaxed) < state_stopping;

            if (HPX_LIKELY(thrd ||
                    scheduler.SchedulingPolicy::get_next_thread(
                        num_thread, running, idle_loop_count, thrd)))
            {
                tfunc_time_wrapper tfunc_time_collector(idle_rate);

                idle_loop_count = 0;
                ++busy_loop_count;

                may_exit = false;

                // Only pending HPX threads will be executed.
                // Any non-pending HPX threads are leftovers from a set_state()
                // call for a previously pending HPX thread (see comments above).
                thread_state state = thrd->get_state();
                thread_state_enum state_val = state.state();

                detail::write_old_state_log(num_thread, thrd, state_val);

                if (HPX_LIKELY(pending == state_val))
                {
                    // switch the state of the thread to active and back to
                    // what the thread reports as its return value

                    {
                        // tries to set state to active (only if state is still
                        // the same as 'state')
                        detail::switch_status thrd_stat (thrd, state);
                        if (HPX_LIKELY(thrd_stat.is_valid() &&
                                thrd_stat.get_previous() == pending))
                        {
                            tfunc_time_wrapper tfunc_time_collector(idle_rate);

                            // thread returns new required state
                            // store the returned state in the thread
                            {
                                is_active_wrapper utilization(counters.is_active_);
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
                                util::itt::caller_context cctx(ctx);
//                                 util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task(domain, thrd->get_description());
                                task.add_metadata(task_id, thrd);
                                task.add_metadata(task_phase, thrd->get_thread_phase());
#endif
                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                exec_time_wrapper exec_time_collector(idle_rate);

#if defined(HPX_HAVE_APEX)
                                util::apex_wrapper apex_profiler(
                                    thrd->get_description());

                                thrd_stat = (*thrd)();

                                if (thrd_stat.get_previous() == terminated)
                                {
                                    apex_profiler.stop();
                                }
                                else
                                {
                                    apex_profiler.yield();
                                }
#else
                                thrd_stat = (*thrd)();
#endif
                            }

#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                            ++counters.executed_thread_phases_;
#endif
                        }
                        else {
                            // some other worker-thread got in between and started
                            // executing this HPX-thread, we just continue with
                            // the next one
                            thrd_stat.disable_restore();
                            detail::write_new_state_log_warning(
                                num_thread, thrd, state_val, "no execution");
                            continue;
                        }

                        // store and retrieve the new state in the thread
                        if (HPX_UNLIKELY(!thrd_stat.store_state(state))) {
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
                    if (HPX_UNLIKELY(state_val == pending))
                    {
                        if (HPX_LIKELY(next_thrd == nullptr)) {
                            // schedule other work
                            scheduler.SchedulingPolicy::wait_or_add_new(
                                num_thread, running, idle_loop_count);
                        }

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        scheduler.SchedulingPolicy::schedule_thread_last(thrd,
                            num_thread);
                        scheduler.SchedulingPolicy::do_some_work(num_thread);
                    }
                    else if (HPX_UNLIKELY(state_val == pending_boost))
                    {
                        thrd->set_state(pending);

                        if (HPX_LIKELY(next_thrd == nullptr))
                        {
                            // reschedule this thread right away if the
                            // background work will be triggered
                            if (HPX_UNLIKELY(
                                busy_loop_count > params.max_busy_loop_count_))
                            {
                                next_thrd = thrd;
                            }
                            else
                            {
                                // schedule other work
                                scheduler.SchedulingPolicy::wait_or_add_new(
                                    num_thread, running, idle_loop_count);

                                // schedule this thread again immediately with
                                // boosted priority
                                scheduler.SchedulingPolicy::schedule_thread(
                                    thrd, num_thread, thread_priority_boost);
                                scheduler.SchedulingPolicy::do_some_work(
                                    num_thread);
                            }
                        }
                        else if (HPX_LIKELY(next_thrd != thrd))
                        {
                            // schedule this thread again immediately with
                            // boosted priority
                            scheduler.SchedulingPolicy::schedule_thread(
                                thrd, num_thread, thread_priority_boost);
                            scheduler.SchedulingPolicy::do_some_work(
                                num_thread);
                        }
                    }
                }
                else if (HPX_UNLIKELY(active == state_val)) {
                    LTM_(warning) << "tfunc(" << num_thread << "): " //-V128
                        "thread(" << thrd->get_thread_id().get() << "), "
                        "description(" << thrd->get_description() << "), "
                        "rescheduling";

                    // re-schedule thread, if it is still marked as active
                    // this might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset
                    // yet
                    //
                    // REVIEW: Passing a specific target thread may set off
                    // the round robin queuing.
                    scheduler.SchedulingPolicy::schedule_thread(thrd, num_thread);
                }

                // Remove the mapping from thread_map_ if HPX thread is depleted
                // or terminated, this will delete the HPX thread as all
                // references go out of scope.
                // REVIEW: what has to be done with depleted HPX threads?
                if (HPX_LIKELY(state_val == depleted || state_val == terminated))
                {
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
                    ++counters.executed_threads_;
#endif
                    scheduler.SchedulingPolicy::destroy_thread(thrd, busy_loop_count);
                }
            }

            // if nothing else has to be done either wait or terminate
            else {
                ++idle_loop_count;

                if (scheduler.SchedulingPolicy::wait_or_add_new(
                        num_thread, running, idle_loop_count))
                {
                    // clean up terminated threads one more time before existing
                    if (scheduler.SchedulingPolicy::cleanup_terminated(true))
                    {
                        // if this is an inner scheduler, exit immediately
                        if (!(scheduler.get_scheduler_mode() & policies::delay_exit))
                        {
                            this_state.store(state_stopped);
                            break;
                        }

                        // otherwise, keep idling for some time
                        if (!may_exit)
                            idle_loop_count = 0;
                        may_exit = true;
                    }
                }

                // do background work in parcel layer and in agas
                if (HPX_UNLIKELY(background_thread))
                {
                    thread_result_type background_result = (*background_thread)();
                    (void)background_result;
                    HPX_ASSERT(background_result.second == nullptr);
                }

                // call back into invoking context
                if (!params.inner_.empty())
                    params.inner_();
            }

            // something went badly wrong, give up
            if (HPX_UNLIKELY(this_state.load() == state_terminating))
                break;

            if (busy_loop_count > params.max_busy_loop_count_)
            {
                busy_loop_count = 0;

                // do background work in parcel layer and in agas
                if (HPX_UNLIKELY(background_thread))
                {
                    thread_result_type background_result = (*background_thread)();
                    (void)background_result;
                    HPX_ASSERT(background_result.second == nullptr);
                }
            }
            else if ((scheduler.get_scheduler_mode() & policies::fast_idle_mode) ||
                idle_loop_count > params.max_idle_loop_count_ || may_exit)
            {
                // clean up terminated threads
                if (idle_loop_count > params.max_idle_loop_count_)
                    idle_loop_count = 0;

                // call back into invoking context
                if (!params.outer_.empty())
                    params.outer_();

                // break if we were idling after 'may_exit'
                if (may_exit)
                {
                    if (scheduler.SchedulingPolicy::cleanup_terminated(true))
                    {
                        this_state.store(state_stopped);
                        break;
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
}}}

#endif


