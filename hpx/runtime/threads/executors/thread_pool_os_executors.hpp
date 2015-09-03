//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_OS_POOL_EXECUTORS_AUG_22_2015_0319PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_OS_POOL_EXECUTORS_AUG_22_2015_0319PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/atomic.hpp>
#include <boost/chrono.hpp>

#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT thread_pool_os_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            thread_pool_os_executor(std::size_t num_threads,
                std::string const& affinity_desc = "");
            ~thread_pool_os_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(closure_type && f, char const* description,
                threads::thread_state_enum initial_state, bool run_now,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(
                boost::chrono::steady_clock::time_point const& abs_time,
                closure_type && f, char const* description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(
                boost::chrono::steady_clock::duration const& rel_time,
                closure_type && f, char const* description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Return an estimate of the number of waiting tasks.
            boost::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

            /// Return the mask for processing units the given thread is allowed
            /// to run on.
            mask_type get_pu_mask(topology const& topology,
                std::size_t num_thread) const
            {
                return pool_.get_pu_mask(topology, num_thread);
            }

            /// Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode)
            {
                pool_.set_scheduler_mode(mode);
            }

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;

            static threads::thread_state_enum thread_function_nullary(
                closure_type func);

        private:
            // the scheduler used by this executor
            Scheduler scheduler_;
            std::string executor_name_;
            threads::policies::callback_notifier notifier_;
            threads::detail::thread_pool<Scheduler> pool_;

            std::size_t num_threads_;

            static boost::atomic<std::size_t> os_executor_count_;
            static std::string get_unique_name();

            // protect scheduler initialization
            typedef boost::mutex mutex_type;
            mutable mutex_type mtx_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct HPX_EXPORT local_queue_os_executor
      : public scheduled_executor
    {
        local_queue_os_executor();

        explicit local_queue_os_executor(std::size_t num_threads,
            std::string const& affinity_desc = "");
    };
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct HPX_EXPORT static_queue_os_executor
      : public scheduled_executor
    {
        static_queue_os_executor();

        explicit static_queue_os_executor(std::size_t num_threads,
            std::string const& affinity_desc = "");
    };
#endif

    struct HPX_EXPORT local_priority_queue_os_executor
      : public scheduled_executor
    {
        local_priority_queue_os_executor();

        explicit local_priority_queue_os_executor(std::size_t num_threads,
            std::string const& affinity_desc = "");
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT static_priority_queue_os_executor
      : public scheduled_executor
    {
        static_priority_queue_os_executor();

        explicit static_priority_queue_os_executor(std::size_t num_threads,
            std::string const& affinity_desc = "");
    };
#endif
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

