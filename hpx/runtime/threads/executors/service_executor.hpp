//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_JAN_13_2013_0222PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_SERVICE_EXECUTOR_JAN_13_2013_0222PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/lcos/local/counting_semaphore.hpp>

#include <boost/atomic.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        class HPX_EXPORT service_executor
          : public threads::detail::executor_base
        {
        public:
            service_executor(char const* pool_name);
            ~service_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* description,
                threads::thread_state_enum initial_state, bool run_now,
                error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time abs_time. This call never blocks, and may violate
            // bounds on the executor's queue size.
            void add_at(boost::posix_time::ptime const& abs_time,
                BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* description,
                error_code& ec);

            // Schedule given function for execution in this executor no sooner
            // than time rel_time from now. This call never blocks, and may
            // violate bounds on the executor's queue size.
            void add_after(boost::posix_time::time_duration const& rel_time,
                BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* description,
                error_code& ec);

            // Return an estimate of the number of waiting tasks.
            std::size_t num_pending_closures(error_code& ec) const;

            // helper function
            void add_no_count(HPX_STD_FUNCTION<void()> f);

        private:
            void thread_wrapper(HPX_STD_FUNCTION<void()> const& f);

            util::io_service_pool* pool_;
            boost::atomic<std::size_t> task_count_;
            lcos::local::counting_semaphore shutdown_sem_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct io_pool_executor : public executor
    {
        io_pool_executor()
          : executor(new detail::service_executor("io-pool"))
        {}
    };

    struct parcel_pool_executor : public executor
    {
        parcel_pool_executor()
          : executor(new detail::service_executor("parcel-pool"))
        {}
    };

    struct timer_pool_executor : public executor
    {
        timer_pool_executor()
          : executor(new detail::service_executor("timer-pool"))
        {}
    };

    struct main_pool_executor : public executor
    {
        main_pool_executor()
          : executor(new detail::service_executor("main-pool"))
        {}
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

