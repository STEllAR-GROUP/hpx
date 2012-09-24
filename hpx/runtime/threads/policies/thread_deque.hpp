//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_583D0662_CA9D_4241_805C_93F92D727E6E)
#define HPX_583D0662_CA9D_4241_805C_93F92D727E6E

#include <map>
#include <memory>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/lockfree/fifo.hpp>
#include <hpx/util/lockfree/deque.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/queue_helpers.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/atomic.hpp>
#include <boost/ptr_container/ptr_map.hpp>

// TODO: add branch prediction and function heat

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{

typedef boost::lockfree::deque<thread_data*> work_item_deque_type;

template <typename Queue, typename Value>
inline void enqueue(Queue& workqueue, Value val)
{ workqueue.push_left(val); }

template <typename Queue, typename Value>
inline void enqueue_last(Queue& workqueue, Value val)
{ workqueue.push_right(val); }

template <typename Queue, typename Value>
inline bool dequeue(Queue& workqueue, Value& val)
{ return workqueue.pop_left(val); }

template <typename Queue, typename Value>
inline bool steal(Queue& workqueue, Value& val)
{ return workqueue.pop_right(val); }

///////////////////////////////////////////////////////////////////////////
struct thread_deque
{
  private:
    // we use a simple mutex to protect the data members for now
    typedef boost::mutex mutex_type;

    // Add this number of threads to the work items queue each time the
    // function \a add_new() is called if the queue is empty.
    enum {
        min_add_new_count = 100,
        max_add_new_count = 100,
        max_delete_count = 100
    };

    // this is the type of the queues of new or pending threads
    typedef work_item_deque_type work_items_type;

    // this is the type of a map holding all threads (except depleted ones)
    typedef boost::ptr_map<
        thread_id_type, thread_data, std::less<thread_id_type>
    > thread_map_type;

    // this is the type of the queue of new tasks not yet converted to
    // threads
    typedef HPX_STD_TUPLE<thread_init_data, thread_state_enum> task_description;

    typedef boost::lockfree::deque<task_description*> task_items_type;

    typedef boost::lockfree::fifo<thread_id_type> thread_id_queue_type;

  protected:
    // add new threads if there is some amount of work available
    std::size_t add_new(boost::int64_t add_count)
    {
        if (0 == add_count)
            return 0;

        std::size_t added = 0;
        task_description* task = 0;

        while (add_count-- && dequeue(new_tasks_, task))
        {
            --new_tasks_count_;

            // measure thread creation time
            util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

            // create the new thread
            thread_state_enum state = HPX_STD_GET(1, *task);
            HPX_STD_UNIQUE_PTR<threads::thread_data> thrd(
                new (memory_pool_) threads::thread_data(
                    HPX_STD_GET(0, *task), memory_pool_, state));

            delete task;

            // add the new entry to the map of all threads
            thread_id_type id = thrd->get_thread_id();
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(id, thrd.get());

            if (!p.second) {
                HPX_THROW_EXCEPTION(hpx::out_of_memory,
                    "threadmanager::add_new",
                    "Couldn't add new thread to the map of threads");
                return 0;
            }

            // only insert the thread into the work-items queue if it is in
            // pending state
            if (state == pending) {
                // pushing the new thread into the pending queue of the
                // specified thread_queue
                ++added;
                schedule_thread(thrd.get());
            }

            // transfer ownership to map
            thrd.release();
        }

        if (added)
        { LTM_(debug) << "add_new: added " << added << " tasks to queues"; }

        return added;
    }

    // steal new threads if there is some amount of work available
    std::size_t steal_new(boost::int64_t add_count, thread_deque* addfrom)
    {
        if (0 == add_count)
            return 0;

        std::size_t added = 0;
        task_description* task = 0;

        while (add_count-- && steal(addfrom->new_tasks_, task))
        {
            --addfrom->new_tasks_count_;

            // measure thread creation time
            util::block_profiler_wrapper<add_new_tag> bp(add_new_logger_);

            // create the new thread
            thread_state_enum state = HPX_STD_GET(1, *task);
            HPX_STD_UNIQUE_PTR<threads::thread_data> thrd(
                new (memory_pool_) threads::thread_data(
                    HPX_STD_GET(0, *task), memory_pool_, state));

            delete task;

            // add the new entry to the map of all threads
            thread_id_type id = thrd->get_thread_id();
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(id, thrd.get());

            if (!p.second) {
                HPX_THROW_EXCEPTION(hpx::out_of_memory,
                    "threadmanager::add_new",
                    "Couldn't add new thread to the map of threads");
                return 0;
            }

            // only insert the thread into the work-items queue if it is in
            // pending state
            if (state == pending) {
                // pushing the new thread into the pending queue of the
                // specified thread_queue
                ++added;
                schedule_thread(thrd.get());
            }

            // transfer ownership to map
            thrd.release();
        }

        if (added)
        { LTM_(debug) << "add_new: added " << added << " tasks to queues"; }

        return added;
    }

    boost::int64_t compute_count()
    {
        // create new threads from pending tasks (if appropriate)
        boost::int64_t add_count = -1; // default is no constraint

        // if we are desperate (no work in the queues), add some even if the
        // map holds more than max_count
        if (max_count_) {
            std::size_t count = thread_map_.size();
            if (max_count_ >= count + min_add_new_count) {
                add_count = max_count_ - count;
                if (add_count < min_add_new_count)
                    add_count = min_add_new_count;
                if (add_count > max_add_new_count)
                    add_count = max_add_new_count;
            }
            else if (work_items_.empty()) {
                add_count = min_add_new_count;    // add this number of threads
                max_count_ += min_add_new_count;  // increase max_count
            }
            else {
                LTM_(debug) << "entered";
                add_count = 0;
            }
        }

        return add_count;
    }

    // This function makes sure all threads which are marked for deletion
    // (state is terminated) are properly destroyed
    bool cleanup_terminated_locked(bool delete_all = false)
    {
        {
            // delete only this many threads
            boost::int64_t delete_count = 
                (std::max)(terminated_items_count_ / 10, 
                    static_cast<boost::int64_t>(max_delete_count));
            thread_id_type todelete;
            while ((delete_all || delete_count) && 
                terminated_items_.dequeue(todelete))
            {
                --terminated_items_count_;
                if (thread_map_.erase(todelete))
                    --delete_count;
            }
        }
        return thread_map_.empty();
    }
  public:
    bool cleanup_terminated()
    {
        mutex_type::scoped_lock lk(mtx_);
        return cleanup_terminated_locked();
    }

    // The maximum number of active threads this thread manager should
    // create. This number will be a constraint only as long as the work
    // items queue is not empty. Otherwise the number of active threads
    // will be incremented in steps equal to the \a min_add_new_count
    // specified above.
    enum { max_thread_count = 1000 };

    thread_deque(std::size_t max_count = max_thread_count)
      : work_items_(128),
        work_items_count_(0),
        terminated_items_(128),
        terminated_items_count_(0),
        max_count_((0 == max_count)
                  ? static_cast<std::size_t>(max_thread_count)
                  : max_count),
        new_tasks_(128),
        new_tasks_count_(0),
        memory_pool_(64),
        add_new_logger_("thread_deque::add_new")
    {}

    void set_max_count(std::size_t max_count = max_thread_count)
    {
        max_count_ = (0 == max_count)
                   ? static_cast<std::size_t>(max_thread_count)
                   : max_count;
    }

    // This returns the current length of the queues (work items and new items)
    boost::int64_t get_queue_length() const
    {
        return work_items_count_ + new_tasks_count_;
    }
    // This returns the current length of the work queue
    boost::int64_t get_work_length() const
    {
        return work_items_count_;
    }
    // This returns the current length of the work queue
    boost::int64_t get_task_length() const
    {
        return new_tasks_count_;
    }

    // create a new thread and schedule it if the initial state is equal to
    // pending
    thread_id_type create_thread(thread_init_data& data,
        thread_state_enum initial_state, bool run_now, error_code& ec)
    {
        if (run_now) {
            mutex_type::scoped_lock lk(mtx_);

            HPX_STD_UNIQUE_PTR<threads::thread_data> thrd(
                new (memory_pool_) threads::thread_data(
                    data, memory_pool_, initial_state));

            // add a new entry in the map for this thread
            thread_id_type id = thrd->get_thread_id();
            std::pair<thread_map_type::iterator, bool> p =
                thread_map_.insert(id, thrd.get());

            if (!p.second) {
                HPX_THROWS_IF(ec, hpx::out_of_memory,
                    "threadmanager::register_thread",
                    "Couldn't add new thread to the map of threads");
                return invalid_thread_id;
            }

            // push the new thread in the pending queue thread
            if (initial_state == pending)
                schedule_thread(thrd.get());

            thrd.release(); // release ownership to the map

            if (&ec != &throws)
                ec = make_success_code();

            // return the thread_id of the newly created thread
            return id;
        }

        // do not execute the work, but register a task description for
        // later thread creation
        enqueue(new_tasks_,
            new task_description(boost::move(data), initial_state));
        ++new_tasks_count_;

        if (&ec != &throws)
            ec = make_success_code();

        return invalid_thread_id; // thread has not been created yet
    }

    bool get_next_thread(threads::thread_data*& thrd)
    {
        if (dequeue(work_items_, thrd)) {
            --work_items_count_;
            return true;
        }
        return false;
    }

    bool steal_next_thread(threads::thread_data*& thrd)
    {
        if (steal(work_items_, thrd)) {
            --work_items_count_;
            return true;
        }
        return false;
    }

    // Schedule the passed thread
    void schedule_thread(threads::thread_data* thrd)
    {
        enqueue(work_items_, thrd);
        ++work_items_count_;
    }

    void schedule_thread_last(threads::thread_data* thrd)
    {
        enqueue_last(work_items_, thrd);
        ++work_items_count_;
    }

    // Destroy the passed thread as it has been terminated
    bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
    {
        if (thrd->is_created_from(&memory_pool_)) {
            thread_id_type id = thrd->get_thread_id();
            terminated_items_.enqueue(id);
            if (++terminated_items_count_ > busy_count / 10)
                cleanup_terminated();
            return true;
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////
    /// Return the number of existing threads with the given state.
    boost::int64_t get_thread_count(thread_state_enum state = unknown) const
    {
        mutex_type::scoped_lock lk(mtx_);
        if (unknown == state)
            return thread_map_.size();

        boost::int64_t num_threads = 0;
        thread_map_type::const_iterator end = thread_map_.end();
        for (thread_map_type::const_iterator it = thread_map_.begin();
             it != end; ++it)
        {
            if ((*it).second->get_state() == state)
                ++num_threads;
        }
        return num_threads;
    }

    ///////////////////////////////////////////////////////////////////////////
    void abort_all_suspended_threads(std::size_t /*num_thread*/)
    {
        mutex_type::scoped_lock lk(mtx_);
        thread_map_type::iterator end =  thread_map_.end();
        for (thread_map_type::iterator it = thread_map_.begin();
              it != end; ++it)
        {
            if ((*it).second->get_state() == suspended) {
                (*it).second->set_state_ex(wait_abort);
                (*it).second->set_state(pending);
                schedule_thread((*it).second);
            }
        }
    }

    bool add_new_or_terminate(std::size_t num_thread, bool running,
                              std::size_t& added)
    {
        if (0 == work_items_count_.load(boost::memory_order_relaxed)) {
            util::try_lock_wrapper<mutex_type> lk(mtx_);
            if (!lk)
                return false;

            // this thread acquired the lock, do maintenance and finally
            // call wait() if no work is available
//            LTM_(info) << "tfunc(" << num_thread << "): queues empty"
//                       << ", threads left: " << thread_map_.size();

            std::size_t addednew = add_new(compute_count());
            added += addednew;

            // stop running after all PX threads have been terminated
            if (added == 0) {
                // Before exiting each of the OS threads deletes the
                // remaining terminated PX threads
                bool canexit = cleanup_terminated_locked(true);
                if (!running && canexit) 
                    return true;

//                 LTM_(debug) << "tfunc(" << num_thread
//                             << "): threadmap not empty";
            }
            return false;
        }

        cleanup_terminated_locked();
        return false;
    }

    bool steal_new_or_terminate(std::size_t num_thread, bool running,
                                std::size_t& added, thread_deque* addfrom)
    {
        if (0 == work_items_count_.load(boost::memory_order_relaxed)) {
            util::try_lock_wrapper<mutex_type> lk(mtx_);
            if (!lk)
                return false;

            // this thread acquired the lock, do maintenance and finally
            // call wait() if no work is available
//            LTM_(debug) << "tfunc(" << num_thread << "): queues empty"
//                        << ", threads left: " << thread_map_.size();

            std::size_t addednew = steal_new(compute_count(), addfrom);
            added += addednew;

            // stop running after all PX threads have been terminated
            if (!(added != 0) && !running) {
                // Before exiting each of the OS threads deletes the
                // remaining terminated PX threads
                if (cleanup_terminated_locked())
                    return true;

                LTM_(debug) << "tfunc(" << num_thread
                           << "): threadmap not empty";
            }

            else {
                cleanup_terminated_locked();
                return false;
            }
        }

        return false;
    }

    // no-op for local scheduling
    void do_some_work() { }

    ///////////////////////////////////////////////////////////////////////
    bool dump_suspended_threads(std::size_t num_thread
      , boost::int64_t& idle_loop_count, bool running)
    {
#if !HPX_THREAD_MINIMAL_DEADLOCK_DETECTION
        return false;
#else
        mutex_type::scoped_lock lk(mtx_);
        return detail::dump_suspended_threads(num_thread, thread_map_
          , idle_loop_count, running);
#endif
    }

    ///////////////////////////////////////////////////////////////////////
    void on_start_thread(std::size_t num_thread) {}
    void on_stop_thread(std::size_t num_thread)
    {
        if (0 == num_thread) {
            // print queue statistics
            log_fifo_statistics(work_items_, "thread_deque");
            log_fifo_statistics(terminated_items_, "thread_deque");
            log_fifo_statistics(new_tasks_, "thread_deque");
        }
    }
    void on_error(std::size_t num_thread, boost::exception_ptr const& e) {}

private:
    mutable mutex_type mtx_;            ///< mutex protecting the members

    thread_map_type thread_map_;        ///< mapping of thread id's to PX-threads
    work_items_type work_items_;        ///< list of active work items
    boost::atomic<boost::int64_t> work_items_count_;    ///< count of active work items
    thread_id_queue_type terminated_items_;   ///< list of terminated threads
    boost::atomic<boost::int64_t> terminated_items_count_;    ///< count of terminated items

    std::size_t max_count_;             ///< maximum number of existing PX-threads
    task_items_type new_tasks_;         ///< list of new tasks to run
    boost::atomic<boost::int64_t> new_tasks_count_;     ///< count of new tasks to run

    threads::thread_pool memory_pool_;  ///< OS thread local memory pools for
                                        ///< PX-threads

    util::block_profiler<add_new_tag> add_new_logger_;
};

}}}

#endif // HPX_583D0662_CA9D_4241_805C_93F92D727E6E

