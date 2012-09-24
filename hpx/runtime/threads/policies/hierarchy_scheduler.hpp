//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_HIERARCHY)
#define HPX_THREADMANAGER_SCHEDULING_HIERARCHY

#include <vector>
#include <memory>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/block_profiler.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/thread_queue.hpp>

#include <boost/noncopyable.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The hierarchy_scheduler maintains a tree of queues of work items
    /// (threads). Every OS threads walks that tree to obtain new work
    class hierarchy_scheduler : boost::noncopyable
    {
    private:
        // The maximum number of active threads this thread manager should
        // create. This number will be a constraint only as long as the work
        // items queue is not empty. Otherwise the number of active threads
        // will be incremented in steps equal to the \a min_add_new_count
        // specified above.
        enum { max_thread_count = 1000 };

    public:
        typedef boost::mpl::false_ has_periodic_maintenance;
        // the scheduler type takes two initialization parameters:
        //    the number of queues
        //    the maxcount per queue
        struct init_parameter
        {
            init_parameter()
              : num_queues_(1),
                max_queue_thread_count_(max_thread_count),
                numa_sensitive_(false)
            {}

            init_parameter(std::size_t num_queues, std::size_t arity,
                    std::size_t max_queue_thread_count = max_thread_count,
                    bool numa_sensitive = false)
              : num_queues_(num_queues),
                arity_(arity),
                max_queue_thread_count_(max_queue_thread_count),
                numa_sensitive_(numa_sensitive)
            {}

            init_parameter(std::pair<std::size_t, std::size_t> const& init,
                    bool numa_sensitive = false)
              : num_queues_(init.first),
                max_queue_thread_count_(init.second),
                numa_sensitive_(numa_sensitive)
            {}

            std::size_t num_queues_;
            std::size_t arity_;
            std::size_t max_queue_thread_count_;
            bool numa_sensitive_;
        };
        typedef init_parameter init_parameter_type;

        typedef std::vector<thread_queue<false>*> level_type;
        typedef std::vector<level_type> tree_type;
        tree_type tree;

        struct flag_type
        {
            boost::atomic<bool> v;
            flag_type() { v = false; }
            flag_type(flag_type const & f) { v.store(f.v.load()); }
            flag_type & operator=(flag_type const & f) { v.store(f.v.load()); return *this; }
            flag_type & operator=(bool b) { v.store(b); return *this;}
            bool operator==(bool b) { return v == b; }
            operator bool() { return v; }
        };

        typedef std::vector<flag_type > level_flag_type;
        typedef std::vector<level_flag_type> flag_tree_type;
        flag_tree_type work_flag_tree;
        flag_tree_type task_flag_tree;

        typedef tree_type::size_type size_type;
        typedef tree_type::difference_type difference_type;
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
                      , new thread_queue<false>(max_queue_thread_count)
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
                level.at(i) = new thread_queue<false>(max_queue_thread_count);
                work_flag_tree.back()[i] = false;
                task_flag_tree.back()[i] = false;
            }

            tree.push_back(level);
            //std::cout << "added " << n << " queues\n";
            if(n<d)
            {
                init_tree(1, max_thread_count);
            }
            else if(n%d == 0)
            {
                init_tree(n/d, max_thread_count);
            }
            else
            {
                init_tree(n/d+1, max_thread_count);
            }
        }

        hierarchy_scheduler(init_parameter_type const& init)
            : d(init.arity_),
            numa_sensitive_(init.numa_sensitive_)
        {
            BOOST_ASSERT(init.num_queues_ != 0);
            init_tree(init.num_queues_, init.max_queue_thread_count_);
        }

        ~hierarchy_scheduler()
        {
            BOOST_ASSERT(tree.size());
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    delete tree[i][j];
                }
            }
        }

        bool numa_sensitive() const { return numa_sensitive_; }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return num_thread;
        }

        ///////////////////////////////////////////////////////////////////////
        // Queries the current length of the queues (work items and new items).
        boost::int64_t get_queue_length(std::size_t num_thread = std::size_t(-1)) const
        {
            BOOST_ASSERT(tree.size());
            // Return queue length of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < tree[0].size());
                return tree[0][num_thread]->get_queue_length();
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
            std::size_t num_thread = std::size_t(-1)) const
        {
            BOOST_ASSERT(tree.size());
            // Return thread count of one specific queue.
            if (std::size_t(-1) != num_thread)
            {
                BOOST_ASSERT(num_thread < tree[0].size());
                return tree[0][num_thread]->get_thread_count(state);
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

        ///////////////////////////////////////////////////////////////////////
        void abort_all_suspended_threads()
        {
            BOOST_ASSERT(tree.size());
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    tree[i][j]->abort_all_suspended_threads(j);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////
        bool cleanup_terminated()
        {
            BOOST_ASSERT(tree.size());
            bool empty = true;
            for(size_type i = 0; i < tree.size(); ++i)
            {
                for(size_type j = 0; j < tree[i].size(); ++j)
                {
                    empty = tree[i][j]->cleanup_terminated() && empty;
                }
            }
            return empty;
        }

        ///////////////////////////////////////////////////////////////////////
        // create a new thread and schedule it if the initial state is equal to
        // pending
        thread_id_type create_thread(thread_init_data& data,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(tree.back().size());
            return tree.back()[0]->create_thread(data, initial_state,
                run_now, 0, ec);
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

            BOOST_ASSERT(level > 0);
            BOOST_ASSERT(level < tree.size());
            BOOST_ASSERT(idx < tree.at(level).size());
            BOOST_ASSERT(parent < tree.at(level-1).size());

            thread_queue<false> * tq = tree[level][idx];
            boost::int64_t num = tq->get_work_length();
            thread_queue<false> * dest = tree[level-1][parent];
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
                        #if defined(BOOST_WINDOWS)
                    Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                    }
                }
            }

            dest->move_work_items_from(
                tq
              , tq->get_work_length()/d + 1
              , num_thread
            );
        }

        /// Return the next thread to be executed, return false if non is
        /// available
        bool get_next_thread(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(num_thread < tree[0].size());

            //std::cout << "get next thread " << num_thread << "\n";
            thread_queue<false> * tq = tree[0][num_thread];

            // check if we need to collect new work from parents
            if(tq->get_work_length() == 0)
            {
                transfer_threads(num_thread/d, num_thread, 1, num_thread);
            }

            if(tq->get_next_thread(thrd, num_thread))
                return true;
            return false;
        }

        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority /*priority*/ = thread_priority_normal)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(tree.back().size());
            tree.back()[0]->schedule_thread(thrd, 0);
        }

        void schedule_thread_last(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            schedule_thread(thrd, num_thread, priority);
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
            BOOST_ASSERT(level > 0);
            if(level == tree.size())
                return;

            BOOST_ASSERT(level > 0);
            BOOST_ASSERT(level < tree.size());
            BOOST_ASSERT(idx < tree.at(level).size());
            BOOST_ASSERT(parent < tree.at(level-1).size());

            thread_queue<false> * tq = tree[level][idx];

            boost::int64_t num = tq->get_task_length();
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
                        #if defined(BOOST_WINDOWS)
                    Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                    }
                }
            }

            thread_queue<false> * dest = tree[level-1][parent];
            dest->move_task_items_from(
                tq
              , tq->get_task_length()/d + 1
            );
        }

        /// This is a function which gets called periodically by the thread
        /// manager to allow for maintenance tasks to be executed in the
        /// scheduler. Returns true if the OS thread calling this function
        /// has to be terminated (i.e. no more work has to be done).
        bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(num_thread < tree.at(0).size());
            std::size_t added = 0;

            thread_queue<false> * tq = tree[0][num_thread];
            if(tq->get_task_length() == 0)
            {
                transfer_tasks(num_thread/d, num_thread, 1);
            }

            bool result = tq->wait_or_add_new(
                num_thread, running, idle_loop_count, added);

            return result && 0 == added;
        }

        /// This function gets called by the threadmanager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread)
        {
            BOOST_ASSERT(tree.size());
        }

        ///////////////////////////////////////////////////////////////////////
        void on_start_thread(std::size_t num_thread)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_start_thread(num_thread);
            //queues_[num_thread]->on_start_thread(num_thread);
        }
        void on_stop_thread(std::size_t num_thread)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_stop_thread(num_thread);
        }
        void on_error(std::size_t num_thread, boost::exception_ptr const& e)
        {
            BOOST_ASSERT(tree.size());
            BOOST_ASSERT(num_thread < tree.at(0).size());
            tree.at(0).at(num_thread)->on_error(num_thread, e);
        }

    private:
        bool numa_sensitive_;
    };

}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

