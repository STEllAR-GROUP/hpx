//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_POOL_EXECUTORS_JAN_11_2013_0831PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_POOL_EXECUTORS_JAN_11_2013_0831PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/util/thread_description.hpp>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/atomic.hpp>
#include <boost/chrono.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        template <typename ExecutorImpl>
        class manage_thread_executor;

        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT thread_pool_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            thread_pool_executor(std::size_t max_punits = 1,
                std::size_t min_punits = 1,
                char const* description = "thread_pool_executor");
            ~thread_pool_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(closure_type && f,
                util::thread_description const& description,
                threads::thread_state_enum initial_state, bool run_now,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(
                boost::chrono::steady_clock::time_point const& abs_time,
                closure_type && f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(
                boost::chrono::steady_clock::duration const& rel_time,
                closure_type && f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Return an estimate of the number of waiting tasks.
            boost::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

            /// Return the mask for processing units the given thread is allowed
            /// to run on.
            mask_cref_type get_pu_mask(topology const& topology,
                std::size_t num_thread) const
            {
                return scheduler_.Scheduler::get_pu_mask(topology, num_thread);
            }

            /// Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode)
            {
                scheduler_.set_scheduler_mode(mode);
            }

        protected:
            friend class manage_thread_executor<thread_pool_executor>;

            // Return the requested policy element
            std::size_t get_policy_element(threads::detail::executor_parameter p,
                error_code& ec) const;

            // The function below are used by the resource manager to
            // interact with the scheduler.

            // Return statistics collected by this scheduler
            void get_statistics(executor_statistics& stats, error_code& ec) const;

            // Provide the given processing unit to the scheduler.
            void add_processing_unit(std::size_t virt_core,
                std::size_t thread_num, error_code& ec);

            // Remove the given processing unit from the scheduler.
            void remove_processing_unit(std::size_t thread_num, error_code& ec);

            // give invoking context a chance to catch up with its tasks
            void suspend_back_into_calling_context(std::size_t virt_core);

        private:
            // internal run method
            void run(std::size_t virt_core, std::size_t num_thread);

            threads::thread_state_enum thread_function_nullary(
                closure_type func);

            // the scheduler used by this executor
            Scheduler scheduler_;
            lcos::local::counting_semaphore shutdown_sem_;

            // collect statistics
            boost::atomic<std::size_t> current_concurrency_;
            boost::atomic<std::size_t> max_current_concurrency_;
            boost::atomic<boost::uint64_t> tasks_scheduled_;
            boost::atomic<boost::uint64_t> tasks_completed_;

            // policy elements
            std::size_t const max_punits_;
            std::size_t const min_punits_;

            // resource manager registration
            std::size_t cookie_;

            // store the self reference to the HPX thread running this scheduler
            std::vector<threads::thread_self*> self_;

            // protect scheduler initialization
            typedef lcos::local::spinlock mutex_type;
            mutex_type mtx_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct HPX_EXPORT local_queue_executor : public scheduled_executor
    {
        local_queue_executor();

        explicit local_queue_executor(std::size_t max_punits,
            std::size_t min_punits = 1);
    };
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct HPX_EXPORT static_queue_executor : public scheduled_executor
    {
        static_queue_executor();

        explicit static_queue_executor(std::size_t max_punits,
            std::size_t min_punits = 1);
    };
#endif

#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
    struct HPX_EXPORT throttle_queue_executor : public scheduled_executor
    {
        throttle_queue_executor();

        explicit throttle_queue_executor(std::size_t max_punits,
            std::size_t min_punits = 1);
    };
#endif

    struct HPX_EXPORT local_priority_queue_executor : public scheduled_executor
    {
        local_priority_queue_executor();

        explicit local_priority_queue_executor(std::size_t max_punits,
            std::size_t min_punits = 1);
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT static_priority_queue_executor : public scheduled_executor
    {
        static_priority_queue_executor();

        explicit static_priority_queue_executor(std::size_t max_punits,
            std::size_t min_punits = 1);
    };
#endif
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

