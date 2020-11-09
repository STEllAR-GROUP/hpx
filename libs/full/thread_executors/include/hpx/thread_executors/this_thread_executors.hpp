//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY) &&                        \
    (defined(HPX_HAVE_STATIC_SCHEDULER) ||                                     \
        defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER))

#include <hpx/affinity/affinity_data.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_executors/thread_executor.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors {
    namespace detail {
        //////////////////////////////////////////////////////////////////////
        template <typename ExecutorImpl>
        class manage_thread_executor;

        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT this_thread_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            explicit this_thread_executor(
                char const* description = "this_thread_executor",
                policies::detail::affinity_data const& affinity_data =
                    policies::detail::affinity_data());
            ~this_thread_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(closure_type&& f,
                util::thread_description const& description,
                threads::thread_schedule_state initial_state, bool run_now,
                threads::thread_stacksize stacksize,
                threads::thread_schedule_hint schedulehint,
                error_code& ec) override;

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(std::chrono::steady_clock::time_point const& abs_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec) override;

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(std::chrono::steady_clock::duration const& rel_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec) override;

            // Return an estimate of the number of waiting tasks.
            std::uint64_t num_pending_closures(error_code& ec) const override;

            // Reset internal (round robin) thread distribution scheme
            void reset_thread_distribution() override;

            /// Set the new scheduler mode
            void set_scheduler_mode(
                threads::policies::scheduler_mode mode) override;

        protected:
            friend class manage_thread_executor<this_thread_executor>;

            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p,
                error_code& ec) const override;

            // The function below are used by the resource manager to
            // interact with the scheduler.

            // Return statistics collected by this scheduler
            void get_statistics(
                executor_statistics& stats, error_code& ec) const;

            // Provide the given processing unit to the scheduler.
            void add_processing_unit(
                std::size_t virt_core, std::size_t thread_num, error_code& ec);

            // Remove the given processing unit from the scheduler.
            void remove_processing_unit(std::size_t thread_num, error_code& ec);

            // return the description string of the underlying scheduler
            char const* get_description() const;

            // give invoking context a chance to catch up with its tasks
            void suspend_back_into_calling_context();

        private:
            // internal run method
            void run();

            threads::thread_result_type thread_function_nullary(
                closure_type func);

            // the scheduler used by this executor
            Scheduler scheduler_;
            lcos::local::counting_semaphore shutdown_sem_;

            std::size_t thread_num_;
            std::size_t parent_thread_num_;
            std::size_t orig_thread_num_;

            // collect statistics
            std::atomic<std::uint64_t> tasks_scheduled_;
            std::atomic<std::uint64_t> tasks_completed_;

            // resource manager registration
            std::size_t cookie_;

            // store the self reference to the HPX thread running this scheduler
            threads::thread_self* self_;

            // protect scheduler initialization
            typedef lcos::local::spinlock mutex_type;
            mutex_type mtx_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct HPX_EXPORT this_thread_static_queue_executor
      : public scheduled_executor
    {
        this_thread_static_queue_executor();
    };
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT this_thread_static_priority_queue_executor
      : public scheduled_executor
    {
        this_thread_static_priority_queue_executor();
    };
#endif
}}}    // namespace hpx::threads::executors

#include <hpx/config/warnings_suffix.hpp>

#endif
