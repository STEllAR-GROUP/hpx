//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_CUSTOMIZED_POOL_EXECUTOR
#define HPX_RUNTIME_THREADS_CUSTOMIZED_POOL_EXECUTOR

#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <hpx/runtime/threads/threadmanager_impl.hpp>

#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        class HPX_EXPORT customized_pool_executor
                : public threads::detail::scheduled_executor_base
        {
        public:
            customized_pool_executor(const std::string &pool_name);
            ~customized_pool_executor();

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

            static threads::thread_result_type thread_function_nullary(
                    closure_type func);

        private:

            typedef hpx::threads::detail::thread_pool* pool_type;
            typedef hpx::threads::policies::scheduler_base* scheduler_type;

            // the scheduler used by this executor
            pool_type pool_;
            scheduler_type scheduler_;

            // protect scheduler initialization
            typedef compat::mutex mutex_type;
            mutable mutex_type mtx_;
        };
    } // namespace detail

    ///////////////////////////////////////////////////////////////////////////

    struct HPX_EXPORT customized_pool_executor : public scheduled_executor
    {
        customized_pool_executor(const std::string &pool_name);
    };

}}}


#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_CUSTOMIZED_POOL_EXECUTOR*/
