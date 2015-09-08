//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM)
#define HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/detail/periodic_maintenance.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#if defined(HPX_HAVE_APEX)
#include <hpx/util/apex.hpp>
#endif

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    inline void write_new_state_log_debug(std::size_t num_thread,
        thread_data_base* thrd, thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): " //-V128
            << "thread(" << thrd->get_thread_id().get() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread,
        thread_data_base* thrd, thread_state_enum state, char const* info)
    {
        // log this in any case
        LTM_(warning) << "tfunc(" << num_thread << "): " //-V128
            << "thread(" << thrd->get_thread_id().get() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_old_state_log(std::size_t num_thread,
        thread_data_base* thrd, thread_state_enum state)
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
        switch_status (thread_data_base* t, thread_state prev_state)
          : thread_(t), prev_state_(prev_state),
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
        thread_state operator=(thread_state_enum new_state)
        {
            return prev_state_ = thread_state(new_state, prev_state_.get_tag() + 1);
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute this
        // HPX-thread in the meantime.
        thread_state get_previous() const
        {
            return prev_state_;
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

    private:
        thread_data_base* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        bool need_restore_state_;
    };

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    struct idle_collect_rate
    {
        idle_collect_rate(boost::uint64_t& tfunc_time, boost::uint64_t& exec_time)
          : start_timestamp_(util::hardware::timestamp())
          , tfunc_time_(tfunc_time)
          , exec_time_(exec_time)
        {}

        void collect_exec_time(boost::uint64_t timestamp)
        {
            exec_time_ += util::hardware::timestamp() - timestamp;
        }
        void take_snapshot()
        {
            if (tfunc_time_ == boost::uint64_t(-1))
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

        boost::uint64_t start_timestamp_;

        boost::uint64_t& tfunc_time_;
        boost::uint64_t& exec_time_;
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

        boost::uint64_t timestamp_;
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
        idle_collect_rate(boost::uint64_t&, boost::uint64_t&) {}
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
    struct scheduling_counters
    {
        scheduling_counters(boost::int64_t& executed_threads,
                boost::int64_t& executed_thread_phases,
                boost::uint64_t& tfunc_time, boost::uint64_t& exec_time)
          : executed_threads_(executed_threads),
            executed_thread_phases_(executed_thread_phases),
            tfunc_time_(tfunc_time),
            exec_time_(exec_time)
        {}

        boost::int64_t& executed_threads_;
        boost::int64_t& executed_thread_phases_;
        boost::uint64_t& tfunc_time_;
        boost::uint64_t& exec_time_;
    };

    struct scheduling_callbacks
    {
        typedef util::function_nonser<void()> callback_type;
        typedef util::function_nonser<bool()> background_callback_type;

        explicit scheduling_callbacks(
                callback_type && outer,
                callback_type && inner = callback_type(),
                background_callback_type && background =
                    background_callback_type())
          : outer_(std::move(outer)),
            inner_(std::move(inner)),
            background_(std::move(background))
        {}

        callback_type outer_;
        callback_type inner_;
        background_callback_type background_;
    };

    template <typename SchedulingPolicy>
    void scheduling_loop(std::size_t num_thread, SchedulingPolicy& scheduler,
        scheduling_counters& counters, scheduling_callbacks& callbacks)
    {
        boost::atomic<hpx::state>& this_state = scheduler.get_state(num_thread);

        util::itt::stack_context ctx;        // helper for itt support
        util::itt::domain domain(get_thread_name().data());
//         util::itt::id threadid(domain, this);
        util::itt::frame_context fctx(domain);

        boost::int64_t idle_loop_count = 0;
        boost::int64_t busy_loop_count = 0;

        idle_collect_rate idle_rate(counters.tfunc_time_, counters.exec_time_);
        tfunc_time_wrapper tfunc_time_collector(idle_rate);

        scheduler.SchedulingPolicy::start_periodic_maintenance(this_state);

        // spin for some time after queues have become empty
        bool may_exit = false;

        while (true) {
            // Get the next HPX thread from the queue
            thread_data_base* thrd = NULL;

            if (scheduler.SchedulingPolicy::get_next_thread(
                    num_thread, idle_loop_count, thrd))
            {
                tfunc_time_wrapper tfunc_time_collector(idle_rate);

                idle_loop_count = 0;
                ++busy_loop_count;
                may_exit = false;

                // Only pending HPX threads will be executed.
                // Any non-pending HPX threads are leftovers from a set_state()
                // call for a previously pending HPX thread (see comments above).
                thread_state state = thrd->get_state();
                thread_state_enum state_val = state;

                detail::write_old_state_log(num_thread, thrd, state_val);

                if (pending == state_val) {
                    // switch the state of the thread to active and back to
                    // what the thread reports as its return value

                    {
                        // tries to set state to active (only if state is still
                        // the same as 'state')
                        detail::switch_status thrd_stat (thrd, state);
                        if (thrd_stat.is_valid() && thrd_stat.get_previous() == pending)
                        {
                            tfunc_time_wrapper tfunc_time_collector(idle_rate);

                            // thread returns new required state
                            // store the returned state in the thread
                            {
#ifdef HPX_HAVE_ITTNOTIFY
                                util::itt::caller_context cctx(ctx);
                                util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task(domain, thrd->get_description());
#endif
                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                exec_time_wrapper exec_time_collector(idle_rate);
#if defined(HPX_HAVE_APEX)
                                util::apex_wrapper apex_profiler(thrd->get_description());
#endif
                                thrd_stat = (*thrd)();
#if defined(HPX_HAVE_APEX)
                                thread_state prev_state = thrd_stat.get_previous();
                                thread_state_enum prev_state_enum = prev_state;
                                if(prev_state_enum == terminated) {
                                    apex_profiler.stop();
                                } else {
                                    apex_profiler.yield();
                                }
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
                        if (!thrd_stat.store_state(state)) {
                            // some other worker-thread got in between and changed
                            // the state of this thread, we just continue with
                            // the next one
                            detail::write_new_state_log_warning(
                                num_thread, thrd, state_val, "no state change");
                            continue;
                        }
                        state_val = state;

                        // any exception thrown from the thread will reset its
                        // state at this point
                    }

                    //detail::write_new_state_log_debug(num_thread, thrd,
                    //    state_val, "normal");

                    // Re-add this work item to our list of work items if the HPX
                    // thread should be re-scheduled. If the HPX thread is suspended
                    // now we just keep it in the map of threads.
                    if (state_val == pending) {
                        // schedule other work
                        scheduler.SchedulingPolicy::wait_or_add_new(num_thread,
                            is_running_state(this_state.load()), idle_loop_count);

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        scheduler.SchedulingPolicy::schedule_thread_last(thrd,
                            num_thread);
                        scheduler.SchedulingPolicy::do_some_work(num_thread);
                    }
                }
                else if (active == state_val) {
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
                if (state_val == depleted || state_val == terminated)
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

                if (scheduler.SchedulingPolicy::wait_or_add_new(num_thread,
                        is_running_state(this_state.load()), idle_loop_count))
                {
                    // clean up terminated threads one more time before existing
                    if (scheduler.SchedulingPolicy::cleanup_terminated(true))
                    {
                        // if this is an inner scheduler, exit immediately
                        if (!callbacks.inner_.empty())
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
                if (!callbacks.background_.empty())
                {
                    if (callbacks.background_())
                    idle_loop_count = 0;
                }

                // call back into invoking context
                if (!callbacks.inner_.empty())
                    callbacks.inner_();
            }

            // something went badly wrong, give up
            if (this_state.load() == state_terminating)
                break;

            if (busy_loop_count > HPX_BUSY_LOOP_COUNT_MAX)
            {
                busy_loop_count = 0;

                // do background work in parcel layer and in agas
                if (!callbacks.background_.empty())
                {
                    if (callbacks.background_())
                    idle_loop_count = 0;
                }
            }
            else if (idle_loop_count > HPX_IDLE_LOOP_COUNT_MAX)
            {
                // call back into invoking context
                if (!callbacks.outer_.empty())
                    callbacks.outer_();

                // clean up terminated threads
                idle_loop_count = 0;

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


