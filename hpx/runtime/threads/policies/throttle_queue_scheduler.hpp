//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2014      Allan Porterfield
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_THROTTLE_QUEUE_MAR_15_2011_0926AM)
#define HPX_THREADMANAGER_SCHEDULING_THROTTLE_QUEUE_MAR_15_2011_0926AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THROTTLE_SCHEDULER)

#include <apex_api.hpp>

#include <boost/thread/shared_mutex.hpp>

#include <vector>
#include <memory>
#include <time.h>

#include <hpx/config/warnings_prefix.hpp>

static boost::shared_mutex init_mutex;

/* Notes on building and running Throttling scheduler

BASED ON THE local_priority_queue scheduler.

add -DHPX_HAVE_THROTTLE_SCHEDULER=1 to cmake to include the throttling during
the HPX build.
I haven't tested but the flag to include all schedulers should also work.

APEX also needs to be available (-DTAU_ROOT=... -DHPX_WITH_APEX=1 needed)

To select the throttling scheduler during execution the --hpx:queuing=throttle
needs to be included.

The HPX execution needs to be running on a system with an active RCRdaemon
writing the RCRblackboard.  (currently I know {elo,thumper}.hpc.renci.org work)

The current model is braindead. It checks to see if the energy is above a fixed
value (80W) and reduces the number of active threads to HPX_THROTTLE_MIN
environment variable (12 if not specified) and when the power is below 50 the
number of threads is set to HPX_THROTTLE_MAX (16 if not specified).

The Power cutoffs should also be controlled via environment variable and the
current memory concurrency should play a significant role in deciding whether
to limit the parallel in the system. During high concurrency the speed is
limited by memory bandwidth and the reduction in parallelism should not
significantly reduce execution time (memory is still going to be running flat
out). I may get to these additions this week, I'll try hard to have them in
place before the visit to Oregon.
*/

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    static std::size_t apex_current_desired_active_threads = INT_MAX;

    ///////////////////////////////////////////////////////////////////////////
    /// The throttle_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from.
    template <typename Mutex
            , typename PendingQueuing
            , typename StagedQueuing
            , typename TerminatedQueuing
             >
    class throttle_queue_scheduler
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

        throttle_queue_scheduler(init_parameter_type const& init,
                bool deferred_initialization = true)
          : base_type(init, deferred_initialization),
            apex_current_threads_(init.num_queues_)
        {
            apex_current_desired_active_threads = init.num_queues_;
        }

        virtual ~throttle_queue_scheduler()
        {
            apex::shutdown_throttling();
        }

        static std::string get_scheduler_name()
        {
            return "throttle_queue_scheduler";
        }

        bool throttle(std::size_t num_thread, std::size_t add_thread)
        {
            // check if we should throttle
            std::size_t desired_active_threads = apex::get_thread_cap();
            if (num_thread < desired_active_threads)
                return true;

            // Sleep so that we don't continue using energy repeatedly
            // checking for work.
            static const struct timespec tim { 0, 100000 };
            nanosleep(&tim, nullptr);
            return false;
        }

        /// Return the next thread to be executed, return false if none is
        /// available
        virtual bool get_next_thread(std::size_t num_thread,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd)
        {
            bool ret = throttle(num_thread, apex_current_threads_ <
                apex_current_desired_active_threads); // am I throttled?
            if (!ret) return false;  // throttled --  don't grab any work

            // grab work if available
            return this->base_type::get_next_thread(
                num_thread, idle_loop_count, thrd);
        }

    protected:
        std::size_t apex_current_threads_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

#endif
