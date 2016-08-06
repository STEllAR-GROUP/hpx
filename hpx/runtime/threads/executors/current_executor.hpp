//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_EXECUTORS_CURRENT_EXECUTOR_HPP
#define HPX_RUNTIME_THREADS_EXECUTORS_CURRENT_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>

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
            void add(closure_type&& f, util::thread_description const& desc,
                threads::thread_state_enum initial_state, bool run_now,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(
                util::steady_clock::time_point const& abs_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(
                util::steady_clock::duration const& rel_time,
                closure_type&& f, util::thread_description const& desc,
                threads::thread_stacksize stacksize, error_code& ec);

            // Return an estimate of the number of waiting tasks.
            std::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

            // Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode);

            // Return the runtime status of the underlying scheduler
            hpx::state get_state() const;

            // retrieve executor id
            executor_id get_id() const
            {
                return create_id(reinterpret_cast<std::size_t>(scheduler_base_));
            }

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

#endif /*HPX_RUNTIME_THREADS_EXECUTORS_CURRENT_EXECUTOR_HPP*/
