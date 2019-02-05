//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// FFWD_TODO: Remove all these comments!

#include "scheduler_base.hpp"
#include "thread_queue.hpp"

#if !defined(HPX_THREADMANAGER_SCHEDULING_FFWD_SCHEDULER)
#define HPX_THREADMANAGER_SCHEDULING_FFWD_SCHEDULER


namespace hpx { namespace threads { namespace policies
{
    template <typename Mutex = compat::mutex,
        typename PendingQueuing = lockfree_lifo,
        typename StagedQueuing = lockfree_lifo,
        typename TerminatedQueuing = lockfree_fifo>
    class HPX_EXPORT ffwd_scheduler : public scheduler_base
    {
    public:
        //////////////////////////////////////////////////////////////////////////////////
        typedef thread_queue<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > thread_queue_type;

        //////////////////////////////////////////////////////////////////////////////////
        ffwd_scheduler(std::size_t num_threads) : scheduler_base(num_threads), thread_count(num_threads), max_queue_thread_count_(1000)
        {
            // everything necessary is done in initializer list
        }

        ~ffwd_scheduler() {
            // messages are empty, all threads have been terminated
        }

        //////////////////////////////////////////////////////////////////////////////////
        std::string get_scheduler_name()
        {
            return "ffwd_scheduler";
        }

        void suspend(std::size_t num_thread)
        {
            std::cout << "suspend called" << std::endl;
            HPX_ASSERT(num_thread < suspend_conds_.size());

            states_[num_thread].store(state_sleeping);
            std::unique_lock<pu_mutex_type> l(suspend_mtxs_[num_thread]);
            suspend_conds_[num_thread].wait(l);

            // Only set running if still in state_sleeping. Can be set with
            // non-blocking/locking functions to stopping or terminating, in
            // which case the state is left untouched.
            hpx::state expected = state_sleeping;
            states_[num_thread].compare_exchange_strong(expected, state_running);

            HPX_ASSERT(expected == state_sleeping ||
                expected == state_stopping || expected == state_terminating);
        }

        ////////////////////////////////////////////////////////////////
        bool numa_sensitive() const { return false; }
        bool has_thread_stealing() const { return false; }


        ///////////////////////////////////////////////////////////////
#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        std::uint64_t get_creation_time(bool reset) {return 0;}
        std::uint64_t get_cleanup_time(bool reset) {return 0;}
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        std::int64_t get_num_pending_misses(std::size_t num_thread,
            bool reset) {return 0;}
        std::int64_t get_num_pending_accesses(std::size_t num_thread,
            bool reset) {return 0;}

        std::int64_t get_num_stolen_from_pending(std::size_t num_thread,
            bool reset) {return 0;}
        std::int64_t get_num_stolen_to_pending(std::size_t num_thread,
            bool reset) {return 0;}
        std::int64_t get_num_stolen_from_staged(std::size_t num_thread,
            bool reset) {return 0;}
        std::int64_t get_num_stolen_to_staged(std::size_t num_thread,
            bool reset) {return 0;}
#endif

        std::int64_t get_queue_length(
            std::size_t num_thread = std::size_t(-1)) const {

            return messages.get_queue_length();
        }

        std::int64_t get_thread_count(
            thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1),
            bool reset = false) const {

            return messages.get_thread_count();
        }

        // Enumerate all matching threads
        bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
                thread_state_enum state = unknown) const {
            std::cout << "enumerate threads not implemented yet" << std::endl;
            return true;
        }

        void abort_all_suspended_threads() {
            std::cout << "abort_all_suspended_threads not implemented yet" << std::endl;
        }

        bool cleanup_terminated(bool delete_all) {

            bool empty = true;
            messages.cleanup_terminated(delete_all);
            return empty;
        }

        bool cleanup_terminated(std::size_t num_thread, bool delete_all) {
            return messages.cleanup_terminated(delete_all);

        }

        void create_thread(thread_init_data& data, thread_id_type* id,
                                   thread_state_enum initial_state, bool run_now, error_code& ec) {
//            {
//                // Entire block was used in earlier implementations to select which thread queue was used
//                // Is it still necessary?
//                std::size_t num_thread =
//                    data.schedulehint.mode == thread_schedule_hint_mode_thread ?
//                    data.schedulehint.hint : std::size_t(-1);
//                std::size_t queue_size = thread_count;

//                if (std::size_t(-1) == num_thread)
//                {
//                    num_thread = curr_queue_++ % queue_size;
//                }
//                else if (num_thread >= queue_size)
//                {
//                    num_thread %= queue_size;
//                }

//                std::unique_lock<pu_mutex_type> l;
//                num_thread = select_active_pu(l, num_thread);
//            }

            messages.create_thread(data, id, initial_state, run_now, ec);
        }

        bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd){

            // we only have our local queue right now
            bool result = messages.get_next_thread(thrd);

            messages.increment_num_pending_accesses();
            if (result) {
                return true;
            }
            messages.increment_num_pending_misses();

            bool have_staged = messages.
                get_staged_queue_length(std::memory_order_relaxed) != 0;

            // Give up, we should have work to convert.
            if (have_staged)
                return false;

            if (!running)
            {
                return false;
            }

            return false;
        }

        void schedule_thread(threads::thread_data* thrd,
            threads::thread_schedule_hint schedulehint,
            bool allow_fallback = false,
            thread_priority priority = thread_priority_normal){
//            {
//                // Entire block was used in earlier implementations to select which thread queue was used
//                // Is it still necessary?
//                // NOTE: This scheduler ignores NUMA hints.
//                std::size_t num_thread = std::size_t(-1);
//                if (schedulehint.mode == thread_schedule_hint_mode_thread)
//                {
//                    num_thread = schedulehint.hint;
//                }
//                else
//                {
//                    allow_fallback = false;
//                }

//                std::size_t queue_size = thread_count;

//                if (std::size_t(-1) == num_thread)
//                {
//                    num_thread = curr_queue_++ % queue_size;
//                }
//                else if (num_thread >= queue_size)
//                {
//                    num_thread %= queue_size;
//                }

//                std::unique_lock<pu_mutex_type> l;
//                num_thread = select_active_pu(l, num_thread, allow_fallback);
//            }

            HPX_ASSERT(thrd->get_scheduler_base() == this);
            messages.schedule_thread(thrd);
        }

        void schedule_thread_last(threads::thread_data* thrd,
            threads::thread_schedule_hint schedulehint,
            bool allow_fallback = false,
                                  thread_priority priority = thread_priority_normal) {

            // NOTE: This scheduler ignores NUMA hints.
            std::size_t num_thread = std::size_t(-1);
            if (schedulehint.mode == thread_schedule_hint_mode_thread)
            {
                num_thread = schedulehint.hint;
            }
            else
            {
                allow_fallback = false;
            }

            std::size_t queue_size = thread_count;

            if (std::size_t(-1) == num_thread)
            {
                num_thread = curr_queue_++ % queue_size;
            }
            else if (num_thread >= queue_size)
            {
                num_thread %= queue_size;
            }

            std::unique_lock<pu_mutex_type> l;
            num_thread = select_active_pu(l, num_thread, allow_fallback);

            HPX_ASSERT(thrd->get_scheduler_base() == this);

            messages.schedule_thread(thrd, true);
        }

        void destroy_thread(threads::thread_data* thrd,
                            std::int64_t& busy_count) {
            HPX_ASSERT(thrd->get_scheduler_base() == this);
            thrd->get_queue<thread_queue_type>().destroy_thread(thrd, busy_count);
        }

        bool wait_or_add_new(std::size_t num_thread, bool running,
                             std::int64_t& idle_loop_count) {
            std::size_t added = 0;
            bool result = true;

            result = messages.wait_or_add_new(running,
                idle_loop_count, added);
            if (0 != added) {
                return result;
            }

            // Check if we have been disabled
            if (!running)
            {
                return true;
            }

            // nothing was found
            return false;
        }

        void on_start_thread(std::size_t num_thread) override {
            messages.on_start_thread(num_thread);
        }
        void on_stop_thread(std::size_t num_thread) override {
            messages.on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread,
            std::exception_ptr const& e) override {
            messages.on_error(num_thread, e);
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        virtual std::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual std::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
#endif

        void start_periodic_maintenance(
            std::atomic<hpx::state>& /*global_state*/)
        {
        }

        void reset_thread_distribution() {
            std::cout << "reset_thread_distribution not implemented yet" << std::endl;
        }

    private:
        std::size_t thread_count;
        std::size_t max_queue_thread_count_;

        thread_queue_type messages;
        std::atomic<std::size_t> curr_queue_;

    };
}}}

#endif
