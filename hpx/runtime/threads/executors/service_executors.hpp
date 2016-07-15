//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP
#define HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/atomic.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        class HPX_EXPORT service_executor
          : public threads::detail::scheduled_executor_base
        {
        public:
            service_executor(char const* pool_name,
                char const* pool_name_suffix = "");
            ~service_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(closure_type&& f,
                util::thread_description const& description,
                threads::thread_state_enum initial_state, bool run_now,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(
                std::chrono::steady_clock::time_point const& abs_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(
                std::chrono::steady_clock::duration const& rel_time,
                closure_type&& f, util::thread_description const& description,
                threads::thread_stacksize stacksize, error_code& ec);

            // Return an estimate of the number of waiting tasks.
            std::uint64_t num_pending_closures(error_code& ec) const;

            // helper functions
            void add_no_count(closure_type&& f);
            void thread_wrapper(closure_type&& f);

        protected:
            // Return the requested policy element
            std::size_t get_policy_element(
                threads::detail::executor_parameter p, error_code& ec) const;

        private:
            util::io_service_pool* pool_;
            boost::atomic<std::uint64_t> task_count_;
            lcos::local::counting_semaphore shutdown_sem_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The type of the HPX thread pool to use for a given service_executor
    ///
    /// This enum type allows to specify the kind of the HPX thread pool to use
    /// for a given \a service_executor.
    enum class service_executor_type
    {
        io_thread_pool,        ///< Selects creating a service executor using
                               ///< the I/O pool of threads
        parcel_thread_pool,    ///< Selects creating a service executor using
                               ///< the parcel pool of threads
        timer_thread_pool,     ///< Selects creating a service executor using
                               ///< the timer pool of threads
        main_thread            ///< Selects creating a service executor using
                               ///< the main thread
    };

    namespace detail
    {
        /// \cond NOINTERNAL
        inline threads::detail::scheduled_executor_base*
        get_service_executor(service_executor_type t,
            char const* name_suffix = "")
        {
            switch(t)
            {
            case service_executor_type::io_thread_pool:
                return new detail::service_executor("io-pool");

            case service_executor_type::parcel_thread_pool:
                {
                    char const* suffix = *name_suffix ? name_suffix : "-tcp";
                    return new detail::service_executor("parcel-pool", suffix);
                }

            case service_executor_type::timer_thread_pool:
                return new detail::service_executor("timer-pool");

            case service_executor_type::main_thread:
                return new detail::service_executor("main-pool");

            default:
                break;
            }

            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::threads::detail::get_service_executor",
                "unknown pool executor type");
            return nullptr;
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    struct service_executor : public scheduled_executor
    {
        service_executor(service_executor_type t,
                char const* name_suffix = "")
          : scheduled_executor(detail::get_service_executor(t, name_suffix))
        {}
    };

    struct io_pool_executor : public scheduled_executor
    {
        io_pool_executor()
          : scheduled_executor(detail::get_service_executor(
                service_executor_type::io_thread_pool))
        {}
    };

    struct parcel_pool_executor : public scheduled_executor
    {
        parcel_pool_executor(char const* name_suffix = "-tcp")
          : scheduled_executor(detail::get_service_executor(
                service_executor_type::parcel_thread_pool, name_suffix))
        {}
    };

    struct timer_pool_executor : public scheduled_executor
    {
        timer_pool_executor()
          : scheduled_executor(detail::get_service_executor(
                service_executor_type::timer_thread_pool))
        {}
    };

    struct main_pool_executor : public scheduled_executor
    {
        main_pool_executor()
          : scheduled_executor(detail::get_service_executor(
                service_executor_type::main_thread))
        {}
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_HPP*/
