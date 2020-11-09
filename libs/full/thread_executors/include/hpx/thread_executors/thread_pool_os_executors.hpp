//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY) &&                        \
    defined(HPX_HAVE_THREAD_POOL_OS_EXECUTOR_COMPATIBILITY)
#include <hpx/affinity/affinity_data.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/thread_executors/thread_executor.hpp>
#include <hpx/thread_pools/scheduled_thread_pool.hpp>
#include <hpx/threading_base/callback_notifier.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors {
    namespace detail {
        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT thread_pool_os_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            thread_pool_os_executor(std::size_t num_threads,
                policies::detail::affinity_data const& affinity_data,
                util::optional<policies::callback_notifier> notifier =
                    util::nullopt);
            ~thread_pool_os_executor();

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

            /// Return the mask for processing units the given thread is allowed
            /// to run on.
            mask_cref_type get_pu_mask(topology const& /*topology*/,
                std::size_t num_thread) const override
            {
                return hpx::resource::get_partitioner().get_pu_mask(num_thread);
            }

            /// Set the new scheduler mode
            void set_scheduler_mode(
                threads::policies::scheduler_mode mode) override
            {
                pool_->get_scheduler()->set_scheduler_mode(mode);
            }

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p,
                error_code& ec) const override;

            static threads::thread_result_type thread_function_nullary(
                closure_type func);

        private:
            // the scheduler used by this executor
            Scheduler* scheduler_;
            std::string executor_name_;
            threads::policies::callback_notifier notifier_;
            std::unique_ptr<threads::detail::scheduled_thread_pool<Scheduler>>
                pool_;
            threads::detail::network_background_callback_type
                network_background_callback_;

            threads::thread_pool_init_parameters thread_pool_init_;

            static std::atomic<std::size_t> os_executor_count_;
            static std::string get_unique_name();

            // protect scheduler initialization
            typedef std::mutex mutex_type;
            mutable mutex_type mtx_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    struct HPX_EXPORT local_queue_os_executor : public scheduled_executor
    {
        local_queue_os_executor(std::size_t num_threads,
            policies::detail::affinity_data const& affinity_data,
            util::optional<policies::callback_notifier> notifier =
                util::nullopt);
    };
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    struct HPX_EXPORT static_queue_os_executor : public scheduled_executor
    {
        static_queue_os_executor(std::size_t num_threads,
            policies::detail::affinity_data const& affinity_data,
            util::optional<policies::callback_notifier> notifier =
                util::nullopt);
    };
#endif

    struct HPX_EXPORT local_priority_queue_os_executor
      : public scheduled_executor
    {
        local_priority_queue_os_executor(std::size_t num_threads,
            policies::detail::affinity_data const& affinity_data,
            util::optional<policies::callback_notifier> notifier =
                util::nullopt);
    };

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT static_priority_queue_os_executor
      : public scheduled_executor
    {
        static_priority_queue_os_executor(std::size_t num_threads,
            policies::detail::affinity_data const& affinity_data,
            util::optional<policies::callback_notifier> notifier =
                util::nullopt);
    };
#endif
}}}    // namespace hpx::threads::executors

#include <hpx/config/warnings_suffix.hpp>
#endif
