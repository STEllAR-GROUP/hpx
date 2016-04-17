//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_HIERARCHY)
#define HPX_THREADMANAGER_SCHEDULING_HIERARCHY

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/mpl/bool.hpp>

#include <memory>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The hierarchy_scheduler maintains a tree of queues of work items
    /// (threads). Every OS threads walks that tree to obtain new work
    template <typename Mutex
            , typename PendingQueuing
            , typename StagedQueuing
            , typename TerminatedQueuing
             >
    class hierarchy_scheduler : public scheduler_base
    {
    private:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        // FIXME: this is specified both here, and in thread_queue.
        enum { max_thread_count = 1000 };

    public:
        typedef boost::mpl::false_ has_periodic_maintenance;

        typedef thread_queue<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > thread_queue_type;

        // the scheduler type takes two initialization parameters:
        //    the number of queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter()
              : num_queues_(1),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false),
                description_("hierarchy_scheduler")
            {}

            init_parameter(std::size_t num_queues, std::size_t arity,
                    std::size_t max_queue_thread_count = max_thread_count,
                    bool numa_sensitive = false,
                    char const* description = "hierarchy_scheduler")
              : num_queues_(num_queues),
                arity_(arity),
                max_queue_thread_count_(max_queue_thread_count),
                numa_sensitive_(numa_sensitive),
                description_(description)
            {}

            init_parameter(std::size_t num_queues, char const* description)
              : num_queues_(num_queues),
                arity_(2),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false),
                description_(description)
            {}

            std::size_t num_queues_;
            std::size_t arity_;
            std::size_t max_queue_thread_count_;
            bool numa_sensitive_;
            char const* description_;
        };
        typedef init_parameter init_parameter_type;

        typedef std::vector<thread_queue_type*> level_type;
        typedef std::vector<level_type> tree_type;
        tree_type tree;

        struct flag_type
        {
            boost::atomic<bool> v;
            flag_type() { v = false; }
            flag_type(flag_type const & f) { v.store(f.v.load()); }
            flag_type & operator=(flag_type const & f) { v.store(f.v.load());
               return *this; }
            flag_type & operator=(bool b) { v.store(b); return *this;}
            bool operator==(bool b) { return v == b; }
            operator bool() { return v; }
        };

        typedef std::vector<flag_type > level_flag_type;
        typedef std::vector<level_flag_type> flag_tree_type;
        flag_tree_type work_flag_tree;
        flag_tree_type task_flag_tree;

        typedef typename tree_type::size_type size_type;
        typedef typename tree_type::difference_type difference_type;
        size_type d;

        void init_tree(size_type n, std::size_t max_queue_thread_count)
        {
            //std::cout << "level " << tree.size() << " " << n << " ";
            if(n==0) return;
            if(n==1)
            {
                //std::cout << "added 1 queue\n";
                tree.push_back(
                    level_type(
                        1
                      , new thread_queue_type(max_queue_thread_count)
                    )
                );
                work_flag_tree.push_back(
                    level_flag_type(
                        1
                    )
                );
                work_flag_tree.back()[0] = false;
                task_flag_tree.push_back(
                    level_flag_type(
                        1
                    )
                );
                task_flag_tree.back()[0] = false;
                return;
            }

            level_type level(n);
            work_flag_tree.push_back(level_flag_type(n));
            task_flag_tree.push_back(level_flag_type(n));
            for(size_type i = 0; i < n; ++i)
            {
                level.at(i) = new thread_queue_type(max_queue_thread_count);
                work_flag_tree.back()[i] = false;
                task_flag_tree.back()[i] = false;
            }

            tree.push_back(level);
            //std::cout << "added " << n << " queues\n";
            if(n<d)
            {
                init_tree(1, max_thread_count); //-V106
            }
            else if(n%d == 0)
            {
                init_tree(n/d, max_thread_count); //-V106
            }
            else
            {
                init_tree(n/d+1, max_thread_count); //-V106
            }
        }

        hierarchy_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : scheduler_base(init.num_queues_, init.description_),
            d(init.arity_),
            numa_sensitive_(init.numa_sensitive_)
        {
            HPX_ASSERT(init.num_queues_ != 0);
            init_tree(init.num_queues_, init.max_queue_thread_count_);
        }

        ~hierarchy_scheduler()
        {
            HPX_ASSERT(tree.size());
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    delete tree[i][j];
                }
            }
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        static std::string get_scheduler_name()
        {
            return "hierarchy_scheduler";
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current length of the queues (work items and new items).
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            HPX_ASSERT(tree.size());
            // Return queue length of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                boost::uint64_t queue_length = 0;
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type const& t = tree[i];
                    if (num_thread < t.size())
                    {
                        queue_length += t[num_thread]->get_queue_length();
                    }
                }
                return queue_length;
            }

            // Cumulative queue lengths of all queues.
            boost::int64_t result = 0;
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree.at(i).size(); ++j)
                {
                    result += tree[i][j]->get_queue_length();
                }
            }
            return result;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current thread count of the queues.
        boost::int64_t get_thread_count(thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1), bool reset = false) const
        {
            HPX_ASSERT(tree.size());
            // Return thread count of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                boost::uint64_t thread_count = 0;
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type const& t = tree[i];
                    if (num_thread < t.size())
                    {
                        thread_count += t[num_thread]->get_thread_count(state);
                    }
                }
                return thread_count;
            }

            // Return the cumulative count for all queues.
            boost::int64_t result = 0;
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    result += tree[i][j]->get_thread_count(state);
                }
            }
            return result;
        }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        ///////////////////////////////////////////////////////////////////////
        boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            HPX_ASSERT(tree.size());

            // Return average thread wait time of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                boost::uint64_t wait_time = 0;
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type const& t = tree[i];
                    if (num_thread < t.size())
                    {
                        wait_time += t[num_thread]->
                            get_average_thread_wait_time();
                    }
                }
                return wait_time;
            }

            // Return the cumulative average thread wait time for all queues.
            boost::int64_t result = 0;
            boost::int64_t count = 0;
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    result += tree[i][j]->get_average_thread_wait_time();
                    ++count;
                }
            }

            return count ? (result / count) : 0; //-V609
        }

        boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const
        {
            HPX_ASSERT(tree.size());

            // Return average task wait time of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                boost::uint64_t wait_time = 0;
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type const& t = tree[i];
                    if (num_thread < t.size())
                    {
                        wait_time += t[num_thread]->
                            get_average_task_wait_time();
                    }
                }
                return wait_time;
            }

            // Return the cumulative average task wait time for all queues.
            boost::int64_t result = 0;
            boost::int64_t count = 0;
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    result += tree[i][j]->get_average_task_wait_time();
                    ++count;
                }
            }

            return count ? (result / count) : 0; //-V609
        }
#endif

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        boost::uint64_t get_creation_time(bool reset)
        {
            boost::uint64_t time = 0;

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                for (size_type j = 0; j != t.size(); ++j)
                    time += t[j]->get_creation_time(reset);
            }
            return time;
        }

        boost::uint64_t get_cleanup_time(bool reset)
        {
            boost::uint64_t time = 0;

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                for (size_type j = 0; j != t.size(); ++j)
                    time += t[j]->get_cleanup_time(reset);
            }
            return time;
        }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        boost::int64_t get_num_pending_misses(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_pending_misses = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != t.size(); ++j)
                        num_pending_misses += t[j]->
                            get_num_pending_misses(reset);
                }
                return num_pending_misses;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_pending_misses += t[num_thread]->
                        get_num_pending_misses(reset);
                }
            }
            return num_pending_misses;
        }

        boost::int64_t get_num_pending_accesses(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_pending_accesses = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != t.size(); ++j)
                        num_pending_accesses +=
                            t[j]->get_num_pending_accesses(reset);
                }
                return num_pending_accesses;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_pending_accesses += t[num_thread]->
                        get_num_pending_accesses(reset);
                }
            }
            return num_pending_accesses;
        }

        boost::int64_t get_num_stolen_from_pending(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != t.size(); ++j)
                        num_stolen_threads +=
                            t[j]->get_num_stolen_from_pending(reset);
                }
                return num_stolen_threads;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_stolen_threads += t[num_thread]->
                        get_num_stolen_from_pending(reset);
                }
            }
            return num_stolen_threads;
        }

        boost::int64_t get_num_stolen_to_pending(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != t.size(); ++j)
                        num_stolen_threads +=
                            t[j]->get_num_stolen_to_pending(reset);
                }
                return num_stolen_threads;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_stolen_threads += t[num_thread]->
                        get_num_stolen_to_pending(reset);
                }
            }
            return num_stolen_threads;
        }

        boost::int64_t get_num_stolen_from_staged(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != tree[i].size(); ++j)
                        num_stolen_threads +=
                            t[j]->get_num_stolen_from_staged(reset);
                }
                return num_stolen_threads;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_stolen_threads += t[num_thread]->
                        get_num_stolen_from_staged(reset);
                }
            }
            return num_stolen_threads;
        }

        boost::int64_t get_num_stolen_to_staged(std::size_t num_thread, bool reset)
        {
            boost::int64_t num_stolen_threads = 0;
            if (num_thread == std::size_t(-1))
            {
                for (size_type i = 0; i != tree.size(); ++i)
                {
                    level_type& t = tree[i];
                    for (size_type j = 0; j != t.size(); ++j)
                        num_stolen_threads +=
                            t[j]->get_num_stolen_to_staged(reset);
                }
                return num_stolen_threads;
            }

            for (size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                if (num_thread < t.size())
                {
                    num_stolen_threads += t[num_thread]->
                        get_num_stolen_to_staged(reset);
                }
            }
            return num_stolen_threads;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            HPX_ASSERT(tree.size());
            for(size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                for(size_type j = 0; j != t.size(); ++j)
                {
                    t[j]->abort_all_suspended_threads();
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated(bool delete_all = false)
        {
            HPX_ASSERT(tree.size());
            bool empty = true;
            for(size_type i = 0; i != tree.size(); ++i)
            {
                level_type& t = tree[i];
                for(size_type j = 0; j != t.size(); ++j)
                {
                    empty = t[j]->cleanup_terminated(delete_all) && empty;
                }
            }
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        // TODO: add recycling
        void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(tree.back().size());
            tree.back()[0]->create_thread(data, id, initial_state,
                run_now, ec);
        }

        void transfer_threads(
            size_type idx
          , size_type parent
          , size_type level
          , std::size_t num_thread
        )
        {
            if(level == tree.size())
                return;

            HPX_ASSERT(level > 0);
            HPX_ASSERT(level < tree.size());
            HPX_ASSERT(idx < tree.at(level).size());
            HPX_ASSERT(parent < tree.at(level-1).size());

            thread_queue_type * tq = tree[level][idx];
            boost::int64_t num = tq->get_pending_queue_length();
            thread_queue_type * dest = tree[level-1][parent];
            if(num == 0)
            {
                if(work_flag_tree[level][idx] == false)
                {
                    work_flag_tree[level][idx] = true;
                    transfer_threads(idx/d, idx, level + 1, num_thread);
                    work_flag_tree[level][idx] = false;
                }
                else
                {
                    while(work_flag_tree[level][idx])
                    {
#if defined(HPX_WINDOWS)
                        Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                        sched_yield();
#else
#endif
                    }
                }
            }

            boost::int64_t count = tq->get_pending_queue_length()/d + 1;
            dest->move_work_items_from(tq, count);

            tq->increment_num_stolen_from_pending(std::size_t(count));
            dest->increment_num_stolen_to_pending(std::size_t(count));
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        bool get_next_thread(std::size_t num_thread,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(num_thread < tree[0].size());

            //std::cout << "get next thread " << num_thread << "\n";
            thread_queue_type * tq = tree[0][num_thread];

            // check if we need to collect new work from parents
            if (tq->get_pending_queue_length() == 0)
            {
                transfer_threads(num_thread/d, num_thread, 1, num_thread);
            }

            bool result = tq->get_next_thread(thrd);

            tq->increment_num_pending_accesses();
            if (result)
                return true;
            tq->increment_num_pending_misses();
            return result;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority /*priority*/ = thread_priority_normal)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(tree.back().size());
            tree.back()[0]->schedule_thread(thrd);
        }

        void schedule_thread_last(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(tree.back().size());
            tree.back()[0]->schedule_thread(thrd, true);
        }

        /// Destroy the passed thread as it has been terminated
        bool destroy_thread(threads::thread_data* thrd, boost::int64_t& busy_count)
        {
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    if(tree[i][j]->destroy_thread(thrd, busy_count))
                        return true;
                }
            }
            return false;
        }

        void transfer_tasks(
            size_type idx
          , size_type parent
          , size_type level
        )
        {
            HPX_ASSERT(level > 0);
            if(level == tree.size())
                return;

            HPX_ASSERT(level > 0);
            HPX_ASSERT(level < tree.size());
            HPX_ASSERT(idx < tree.at(level).size());
            HPX_ASSERT(parent < tree.at(level-1).size());

            thread_queue_type * tq = tree[level][idx];

            boost::int64_t num = tq->get_staged_queue_length();
            if(num == 0)
            {
                if(task_flag_tree[level][idx] == false)
                {
                    task_flag_tree[level][idx] = true;
                    transfer_tasks(idx/d, idx, level + 1);
                    task_flag_tree[level][idx] = false;
                }
                else
                {
                    while(task_flag_tree[level][idx])
                    {
#if defined(HPX_WINDOWS)
                    Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                    }
                }
            }

            thread_queue_type * dest = tree[level-1][parent];
            dest->move_task_items_from(
                tq
              , tq->get_staged_queue_length()/d + 1
            );
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(num_thread < tree.at(0).size());
            std::size_t added = 0;

            thread_queue_type * tq = tree[0][num_thread];
            if(tq->get_staged_queue_length() == 0)
            {
                transfer_tasks(num_thread/d, num_thread, 1);
            }

            bool result = tq->wait_or_add_new(running, idle_loop_count, added);
            return result && 0 == added;
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_start_thread(num_thread);
        }
        void on_stop_thread(std::size_t num_thread)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            HPX_ASSERT(tree.size());
            HPX_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_error(num_thread, e);
        }

    private:
        bool numa_sensitive_;
    };

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

