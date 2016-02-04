//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_CURRENT_EXECUTOR_JAN_11_2013_0831PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_CURRENT_EXECUTOR_JAN_11_2013_0831PM

#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>

#include <boost/ptr_container/ptr_vector.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        class HPX_EXPORT current_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            current_executor(policies::scheduler_base* scheduler);

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

            /// Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode);

            // Return the runtime status of the underlying scheduler
            hpx::state get_state() const;

        protected:
            static threads::thread_state_enum thread_function_nullary(
                closure_type func);

            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;

        private:
            policies::scheduler_base* scheduler_base_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT current_executor : public scheduled_executor
    {
        current_executor();
        explicit current_executor(policies::scheduler_base* scheduler);

        hpx::state get_state() const;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

