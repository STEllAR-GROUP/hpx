//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2014      Allan Porterfield
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_THROTTLING_POLICY_APR_14_2017_1447PM)
#define HPX_THREADMANAGER_SCHEDULING_THROTTLING_POLICY_APR_14_2017_1447PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THROTTLING_SCHEDULER) && defined(HPX_HAVE_ALLSCALE)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/runtime/threads_fwd.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <time.h>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

static boost::shared_mutex init_mutex;


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    static std::size_t allscale_current_desired_active_threads = UINT_MAX;

    ///////////////////////////////////////////////////////////////////////////
    /// The throttling_scheduler extends local_queue_scheduler.
    /// The local_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available. 
    template <typename Mutex = boost::mutex,
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
        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::cond_; 
        using local_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing>::curr_queue_;


        throttling_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization),
            allscale_current_threads_(init.num_queues_)
        {
            allscale_current_desired_active_threads = init.num_queues_;
            disabled_os_threads_.resize(hpx::get_os_thread_count());
        }

        virtual ~throttling_scheduler()
        {
            //apex::shutdown_throttling();
        }

        static std::string get_scheduler_name()
        {
            return "throttling_scheduler";
        }


        bool disabled(std::size_t shepherd) {
             return disabled_os_threads_[shepherd];
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        virtual bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
	    /*
		Drain the queue of the suspended OS thread
	    */
            while (disabled(num_thread)) {
                thread_queue_type* q = queues_[num_thread];
	        while (q->get_next_thread(thrd)) {
		      this->wait_or_add_new(num_thread, running, idle_loop_count) ;
        	      this->schedule_thread(thrd, num_thread, thread_priority_unknown); //TODO: what thread_priority should we use
		}

		boost::chrono::milliseconds period(1);
		boost::unique_lock<boost::mutex> l(mtx_);
        	cond_.wait_for(l, period);
            }

            // grab work if available
            return this->base_type::get_next_thread(
                num_thread, running, idle_loop_count, thrd);
        }


        /// Schedule the passed thread
        void schedule_thread(threads::thread_data* thrd, std::size_t num_thread,
            thread_priority priority = thread_priority_normal)
        {
	    // Loop somehow and find a thread that is not disabled
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

        void disable(std::size_t shepherd) 
	{
	        std::cout << "Starting disable. shepherd: " << shepherd << std::endl;
		
		if (disabled(shepherd))
		    return;

	        if (disabled_os_threads_.size() - disabled_os_threads_.count() < 2 ) {
		    std::cout << "Not enough running threads to suspend. size: " << disabled_os_threads_.size() << ", count: " << disabled_os_threads_.count() << std::endl;
		    return;
		}

		std::lock_guard<mutex_type> l(throttle_mtx_);

		if (shepherd >= disabled_os_threads_.size()) {
		    HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttling_scheduler::suspend",
			"invalid thread number");
		}

		if (!disabled(shepherd)) {
		    disabled_os_threads_[shepherd] = true;
		}
 
	}

 	void enable(std::size_t shepherd) 
	{
	        std::lock_guard<mutex_type> l(throttle_mtx_);

		if (shepherd == -1)
		   return;

		if (shepherd >= disabled_os_threads_.size()) {
		    HPX_THROW_EXCEPTION(hpx::bad_parameter, "throttling_scheduler::resume",
			"invalid thread number");
		}

	        disabled_os_threads_[shepherd] = false; 
		cond_.notify_one();
	}


        void enable_one()
   	{
		std::lock_guard<mutex_type> l(throttle_mtx_);
		if (disabled_os_threads_.any()) { 
		   for (int i=0; i<disabled_os_threads_.size(); i++)
		       if (disabled_os_threads_[i]) {
			  disabled_os_threads_[i] = false;
			  cond_.notify_one();
			  std::cout << "Resumed worker_id: " << i << std::endl;
			  break;
		       }
		}
        }


        void enable_all()
        {
                std::lock_guard<mutex_type> l(throttle_mtx_);
                if (disabled_os_threads_.any()) {
                   for (int i=0; i<disabled_os_threads_.size(); i++)
                       if (disabled_os_threads_[i]) {
                          disabled_os_threads_[i] = false;
                          cond_.notify_one();
                          std::cout << "Resumed worker_id: " << i << std::endl;
                       }
                }
        }

	
	boost::dynamic_bitset<> const & get_disabled_os_threads()
	{
		return disabled_os_threads_;
	}

    protected:
	typedef hpx::lcos::local::spinlock mutex_type;
	mutex_type throttle_mtx_;
        std::size_t allscale_current_threads_;
	boost::dynamic_bitset<> disabled_os_threads_;
	boost::mutex mtx_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

#endif
