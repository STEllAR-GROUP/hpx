//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_DEFAULT_EXECUTOR_JAN_11_2013_0700PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_DEFAULT_EXECUTOR_JAN_11_2013_0700PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        class HPX_EXPORT default_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            default_executor();

            default_executor(thread_priority priority,
                thread_stacksize stacksize, std::size_t os_thread);

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
            inline void add_after(
                boost::chrono::steady_clock::duration const& rel_time,
                closure_type && f, char const* description,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                return add_at(boost::chrono::steady_clock::now() + rel_time,
                    std::move(f), description, stacksize, ec);
            }

            // Return an estimate of the number of waiting tasks.
            boost::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

            /// Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode);

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;

        private:
            thread_stacksize stacksize_;
            thread_priority priority_;
            std::size_t os_thread_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct default_executor : public scheduled_executor
    {
        default_executor()
          : scheduled_executor(new detail::default_executor())
        {}

        default_executor(thread_stacksize stacksize)
          : scheduled_executor(new detail::default_executor(
                thread_priority_default, stacksize, std::size_t(-1)))
        {}

        default_executor(thread_priority priority,
                thread_stacksize stacksize = thread_stacksize_default,
                std::size_t os_thread = std::size_t(-1))
          : scheduled_executor(new detail::default_executor(
                priority, stacksize, os_thread))
        {}

        default_executor(std::size_t os_thread)
          : scheduled_executor(new detail::default_executor(
                thread_priority_default, thread_stacksize_default, os_thread))
        {}
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

