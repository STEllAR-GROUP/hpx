//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM)
#define HPX_RUNTIME_THREADS_DETAIL_SCHEDULING_LOOP_JAN_11_2013_0838PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/hardware/timestamp.hpp>

#include <boost/cstdint.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/asio/deadline_timer.hpp>

namespace hpx { namespace threads { namespace detail 
{
    ///////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    inline void periodic_maintenance_handler(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, boost::mpl::false_)
    {
    }

    template <typename SchedulingPolicy>
    inline void periodic_maintenance_handler(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, boost::mpl::true_)
    {
        scheduler.periodic_maintenance(global_state == running);

        if (global_state.load() == running)
        {
            // create timer firing in correspondence with given time
            boost::asio::deadline_timer t (
                get_thread_pool("timer-thread")->get_io_service(),
                boost::posix_time::milliseconds(1000));

            void (*handler)(SchedulingPolicy&, boost::atomic<hpx::state>&, boost::mpl::true_) =
                &periodic_maintenance_handler<SchedulingPolicy>;

            t.async_wait(boost::bind(handler, boost::ref(scheduler), 
                boost::ref(global_state), boost::mpl::true_()));
        }
    }

    template <typename SchedulingPolicy>
    inline void start_periodic_maintenance(SchedulingPolicy&,
        boost::atomic<hpx::state>& global_state, boost::mpl::false_)
    {
    }

    template <typename SchedulingPolicy>
    inline void start_periodic_maintenance(SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, boost::mpl::true_)
    {
        scheduler.periodic_maintenance(global_state == running);

        boost::posix_time::milliseconds expire(1000);

        // create timer firing in correspondence with given time
        boost::asio::deadline_timer t (
            get_thread_pool("io-thread")->get_io_service(),
            boost::posix_time::milliseconds(1000));

        void (*handler)(SchedulingPolicy&, boost::atomic<hpx::state>&, boost::mpl::true_) =
            &periodic_maintenance_handler<SchedulingPolicy>;

        t.async_wait(boost::bind(handler, boost::ref(scheduler), 
            boost::ref(global_state), boost::mpl::true_()));
    }

    ///////////////////////////////////////////////////////////////////////
    inline void write_new_state_log_debug(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state, char const* info)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
            << "thread(" << thrd->get_thread_id() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_new_state_log_warning(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state, char const* info)
    {
        // log this in any case
        LTM_(warning) << "tfunc(" << num_thread << "): "
            << "thread(" << thrd->get_thread_id() << "), "
            << "description(" << thrd->get_description() << "), "
            << "new state(" << get_thread_state_name(state) << "), "
            << info;
    }
    inline void write_old_state_log(std::size_t num_thread,
        thread_data* thrd, thread_state_enum state)
    {
        LTM_(debug) << "tfunc(" << num_thread << "): "
                    << "thread(" << thrd->get_thread_id() << "), "
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
        // PX-thread in the meantime.
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
        thread_data* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        bool need_restore_state_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename SchedulingPolicy>
    void scheduling_loop(std::size_t num_thread, SchedulingPolicy& scheduler,
        boost::atomic<hpx::state>& global_state, boost::int64_t& executed_threads,
        boost::uint64_t& tfunc_time, boost::uint64_t& exec_time)
    {
        util::itt::stack_context ctx;        // helper for itt support
        util::itt::domain domain(get_thread_name()->data());
//         util::itt::id threadid(domain, this);
        util::itt::frame_context fctx(domain);

        boost::int64_t idle_loop_count = 0;
        boost::int64_t busy_loop_count = 0;

        boost::uint64_t overall_timestamp = util::hardware::timestamp();

        typedef typename SchedulingPolicy::has_periodic_maintenance pred;
        detail::start_periodic_maintenance(scheduler, global_state, pred());

        while (true) {
            // Get the next HPX thread from the queue
            thread_data* thrd = NULL;
            if (scheduler.get_next_thread(num_thread,
                    global_state == running, idle_loop_count, thrd))
            {
                idle_loop_count = 0;
                ++busy_loop_count;

                // Only pending PX threads will be executed.
                // Any non-pending PX threads are leftovers from a set_state()
                // call for a previously pending PX thread (see comments above).
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
                        if (thrd_stat.is_valid() && thrd_stat.get_previous() == pending) {
                            // thread returns new required state
                            // store the returned state in the thread
                            {
#if defined(HPX_USE_ITTNOTIFY)
                                util::itt::caller_context cctx(ctx);
                                util::itt::undo_frame_context undoframe(fctx);
                                util::itt::task task(domain, thrd->get_description());
#endif
                                // Record time elapsed in thread changing state
                                // and add to aggregate execution time.
                                boost::uint64_t timestamp = util::hardware::timestamp();
                                thrd_stat = (*thrd)();
                                exec_time += util::hardware::timestamp() - timestamp;
                            }

                            tfunc_time = util::hardware::timestamp() - overall_timestamp;
                            ++executed_threads;
                        }
                        else {
                            // some other worker-thread got in between and started
                            // executing this PX-thread, we just continue with
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

                    detail::write_new_state_log_debug(num_thread, thrd,
                        state_val, "normal");

                    // Re-add this work item to our list of work items if the PX
                    // thread should be re-scheduled. If the PX thread is suspended
                    // now we just keep it in the map of threads.
                    if (state_val == pending) {
                        // schedule other work
                        scheduler.wait_or_add_new(num_thread,
                            global_state == running, idle_loop_count);

                        // schedule this thread again, make sure it ends up at
                        // the end of the queue
                        // REVIEW: Passing a specific target thread may screw
                        // with the round robin queuing.
                        scheduler.schedule_thread_last(thrd, num_thread);
                        scheduler.do_some_work(num_thread);
                    }
                }
                else if (active == state_val) {
                    LTM_(warning) << "tfunc(" << num_thread << "): "
                        "thread(" << thrd->get_thread_id() << "), "
                        "description(" << thrd->get_description() << "), "
                        "rescheduling";

                    // re-schedule thread, if it is still marked as active
                    // this might happen, if some thread has been added to the
                    // scheduler queue already but the state has not been reset
                    // yet
                    // REVIEW: Passing a specific target thread may screw
                    // with the round robin queuing.
                    scheduler.schedule_thread(thrd, num_thread);
                }

                // Remove the mapping from thread_map_ if PX thread is depleted
                // or terminated, this will delete the PX thread as all
                // references go out of scope.
                // REVIEW: what has to be done with depleted PX threads?
                if (state_val == depleted || state_val == terminated)
                    scheduler.destroy_thread(thrd, busy_loop_count);

                tfunc_time = util::hardware::timestamp() - overall_timestamp;
            }

            // if nothing else has to be done either wait or terminate
            else {
                if (scheduler.wait_or_add_new(num_thread,
                        global_state == running, idle_loop_count))
                {
                    break;
                }
            }

            // Clean up all terminated threads for all thread queues once in a
            // while.
            if (busy_loop_count > HPX_BUSY_LOOP_COUNT_MAX) {
                busy_loop_count = 0;
                scheduler.cleanup_terminated(true);
            }
        }

        // after tfunc loop broke, record total time elapsed
        tfunc_time = util::hardware::timestamp() - overall_timestamp;
    }
}}}

#endif


