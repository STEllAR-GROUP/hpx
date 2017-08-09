//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2014      Allan Porterfield
//  Copyright (c) 2017      Khalid Hasanov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_THROTTLING_POLICY_APR_14_2017_1447PM)
#define HPX_THREADMANAGER_SCHEDULING_THROTTLING_POLICY_APR_14_2017_1447PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THROTTLING_SCHEDULER) && defined(HPX_HAVE_HWLOC)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/compat/mutex.hpp>

#include <hwloc.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <time.h>
#include <vector>
#include <thread>

#include <hpx/config/warnings_prefix.hpp>



///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The throttling_scheduler extends local_queue_scheduler.
    /// The local_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    template <typename Mutex = hpx::compat::mutex,
        typename PendingQueuing = lockfree_fifo,
        typename StagedQueuing = lockfree_fifo,
        typename TerminatedQueuing = lockfree_lifo>
    class HPX_EXPORT throttling_scheduler
      : public local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>
    {
    private:
        typedef local_queue_scheduler<
                Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
            > base_type;

    public:
        typedef typename base_type::has_periodic_maintenance
            has_periodic_maintenance;
        typedef typename base_type::thread_queue_type thread_queue_type;
        typedef typename base_type::init_parameter_type init_parameter_type;

        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::queues_;

#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::mtx_;
        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::cond_;
#endif
        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::curr_queue_;


        throttling_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization)
        {
            disabled_os_threads_.resize(hpx::get_os_thread_count());
            init_num_cores();
        }

        virtual ~throttling_scheduler()
        {
            //enable_more(disabled_os_threads_.size());//Does it make sense here?
        }

        static std::string get_scheduler_name()
        {
            return "throttling_scheduler";
        }

        /// Check if the OS thread disabled
        bool disabled(std::size_t shepherd) {
             return disabled_os_threads_[shepherd];
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        virtual bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            //Drain the queue of the disableed OS thread
            while (disabled(num_thread)) {
                thread_queue_type* q = queues_[num_thread];
                while (q->get_next_thread(thrd)) {
                    this->wait_or_add_new(num_thread, running, idle_loop_count) ;
                    this->schedule_thread(thrd, num_thread, thread_priority_normal);
                }

                std::unique_lock<std::mutex> l(mtx_);
                cond_.wait(l);
            }

            // grab work if available
            return this->base_type::get_next_thread(num_thread,
                running, idle_loop_count, thrd);
        }


        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
            // Loop and find a thread that is not disabled
            if (std::size_t(-1) == num_thread)
               num_thread = curr_queue_++ % queues_.size();
            else
            {
                for(std::size_t tid=0; tid<queues_.size(); tid++) {
                    // Loop and find a thread that is not disabled
                    if (!disabled(tid)) {
                        num_thread = tid;
                        break;
                    }
                }
            }

            HPX_ASSERT(num_thread < queues_.size());

            queues_[num_thread]->schedule_thread(thrd);
        }

        /// Inits the number of physical and logical cores
        void init_num_cores()
        {
            hwloc_topology_t topology;
            hwloc_topology_init(&topology);
            hwloc_topology_load(topology);

            int core_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
            HPX_ASSERT(core_depth != HWLOC_TYPE_DEPTH_UNKNOWN);

            num_physical_cores = hwloc_get_nbobjs_by_depth(topology, core_depth);
            num_logical_cores = std::thread::hardware_concurrency();
            HPX_ASSERT(num_logical_cores % num_physical_cores == 0);

            num_hw_threads = num_logical_cores/num_physical_cores;
            //numa_domains = hpx::compute::host::numa_domains()
        }

        /// Disables specific OS thread requested by the user/application
        void disable(std::size_t shepherd)
        {
            std::lock_guard<mutex_type> l(throttle_mtx_);

            if (disabled(shepherd))
                return;

            if (disabled_os_threads_.size() - disabled_os_threads_.count()
                        < 2 ) {
                return;
            }

            const std::size_t wtid = hpx::get_worker_thread_num();

            if (shepherd == wtid)
               return;

            if (shepherd >= disabled_os_threads_.size()) {
                HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttling_scheduler::disable",
                "invalid thread number");
            }

            if (!disabled(shepherd)) {
                disabled_os_threads_[shepherd] = true;
            }
        }

        /// Decides itself which OS threads to disable.
        /// Currently it picks up thread ids sequentially
        void disable_more(std::size_t num_threads = 1)
        {
            std::lock_guard<mutex_type> l(throttle_mtx_);

            if (num_threads >= disabled_os_threads_.size()) {
                 HPX_THROW_EXCEPTION(hpx::bad_parameter,
                     "throttling_scheduler::disable_more",
                     "invalid number of threads");
            }

            /// If we don't have the requested number of available threads return
            if (disabled_os_threads_.size() - disabled_os_threads_.count()
                    < num_threads ) {
                return;
            }

            if (disabled_os_threads_.size() - disabled_os_threads_.count()
                        < 2 ) {
                return;
            }

            std::size_t wtid = hpx::get_worker_thread_num();

            std::size_t cnt = 0;
            std::size_t tid_start = 0;
            while ( cnt < num_threads ) {
                for (std::size_t i=tid_start; i<disabled_os_threads_.size();
                     i += num_hw_threads)
                {
                    if (!disabled_os_threads_[i] && i != wtid) {
                       disabled_os_threads_[i] = true;
                       cnt++;
                       if (cnt == num_threads) return;
                    }
                }
                tid_start++;
            }
        }

        /// Enable specific OS thread requested by the user/application
        void enable(std::size_t shepherd)
        {
            std::lock_guard<mutex_type> l(throttle_mtx_);

            if (shepherd >= disabled_os_threads_.size()) {
                HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttling_scheduler::enable",
                              "invalid thread number");
            }

            disabled_os_threads_[shepherd] = false;
            cond_.notify_one();
        }


        /// Decides itself which OS threads to enable.
        /// Currently it picks up thread ids sequentially
        void enable_more(std::size_t num_threads = 1)
        {
            std::lock_guard<mutex_type> l(throttle_mtx_);

            std::size_t cnt = 0;
            if (disabled_os_threads_.any()) {
                for (std::size_t i=0; i<disabled_os_threads_.size(); i++)
                    if (disabled_os_threads_[i]) {
                        disabled_os_threads_[i] = false;
                        cond_.notify_one();
                        cnt++;
                        //std::cout << "Enabled worker_id: " << i << std::endl;
                        if (cnt == num_threads) break;
                    }
               }
        }

        /// Return the thread bitset
        boost::dynamic_bitset<> const & get_disabled_os_threads()
        {
            return disabled_os_threads_;
        }

    protected:
        typedef hpx::lcos::local::spinlock mutex_type;
        mutex_type throttle_mtx_;
#if !defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        mutable compat::mutex mtx_;
        compat::condition_variable cond_;
#endif
        mutable boost::dynamic_bitset<> disabled_os_threads_;
        int num_physical_cores;
        int num_logical_cores;
        int num_hw_threads;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

#endif
