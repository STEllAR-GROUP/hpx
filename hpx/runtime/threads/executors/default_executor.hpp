//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_EXECUTORS_DEFAULT_EXECUTOR_HPP
#define HPX_RUNTIME_THREADS_EXECUTORS_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>

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
                thread_stacksize stacksize, thread_schedule_hint schedulehint);

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(closure_type&& f,
                threads::thread_schedule_hint schedulehint,
                util::thread_description const& desc,
                threads::thread_state_enum initial_state,
                threads::thread_stacksize stacksize,
                error_code& ec) override;

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(
                util::steady_clock::time_point const& abs_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec) override;

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            inline void add_after(
                util::steady_clock::duration const& rel_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec) override
            {
                return add_at(util::steady_clock::now() + rel_time,
                    std::move(f), description, stacksize, ec);
            }

            // Return an estimate of the number of waiting tasks.
            std::uint64_t num_pending_closures(error_code& ec) const;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution();

            /// Set the new scheduler mode
            void set_scheduler_mode(threads::policies::scheduler_mode mode);

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;
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
                thread_priority_default, stacksize, thread_schedule_hint_none))
        {}

        default_executor(thread_priority priority,
                thread_stacksize stacksize = thread_stacksize_default,
                thread_schedule_hint schedulehint = thread_schedule_hint_none)
          : scheduled_executor(new detail::default_executor(
                priority, stacksize, schedulehint))
        {}

        default_executor(thread_schedule_hint schedulehint)
          : scheduled_executor(new detail::default_executor(
                thread_priority_default, thread_stacksize_default, schedulehint))
        {}
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_EXECUTORS_DEFAULT_EXECUTOR_HPP*/
