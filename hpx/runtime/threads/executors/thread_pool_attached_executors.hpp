//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_HPP
#define HPX_RUNTIME_THREADS_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/atomic.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT thread_pool_attached_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            thread_pool_attached_executor(
                std::size_t first_thread, std::size_t num_threads,
                thread_priority priority, thread_stacksize stacksize);

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
                util::steady_clock::time_point const& abs_time,
                closure_type && f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(
                util::steady_clock::duration const& rel_time,
                closure_type && f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Return an estimate of the number of waiting tasks.
            std::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;

            std::size_t get_next_thread_num()
            {
                return first_thread_ + (os_thread_++ % num_threads_);
            }

        private:
            std::size_t first_thread_;
            std::size_t num_threads_;
            boost::atomic<std::size_t> os_thread_;
            thread_priority priority_;
            thread_stacksize stacksize_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct HPX_EXPORT local_queue_attached_executor
      : public scheduled_executor
    {
        explicit local_queue_attached_executor(
            std::size_t first_thread = 0, std::size_t num_threads = 1,
            thread_priority priority = thread_priority_default,
            thread_stacksize stacksize = thread_stacksize_default);
    };
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct HPX_EXPORT static_queue_attached_executor
      : public scheduled_executor
    {
        explicit static_queue_attached_executor(
            std::size_t first_thread = 0, std::size_t num_threads = 1,
            thread_priority priority = thread_priority_default,
            thread_stacksize stacksize = thread_stacksize_default);
    };
#endif

    struct HPX_EXPORT local_priority_queue_attached_executor
      : public scheduled_executor
    {
        explicit local_priority_queue_attached_executor(
            std::size_t first_thread, std::size_t num_threads,
            thread_priority priority = thread_priority_default,
            thread_stacksize stacksize = thread_stacksize_default);
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT static_priority_queue_attached_executor
      : public scheduled_executor
    {
        explicit static_priority_queue_attached_executor(
            std::size_t first_thread = 0, std::size_t num_threads = 1,
            thread_priority priority = thread_priority_default,
            thread_stacksize stacksize = thread_stacksize_default);
    };
#endif
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_EXECUTORS_THREAD_POOL_ATTACHED_EXECUTORS_HPP*/
