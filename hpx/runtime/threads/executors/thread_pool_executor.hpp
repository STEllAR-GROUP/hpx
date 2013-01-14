//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_SERIAL_EXECUTOR_JAN_11_2013_0831PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_SERIAL_EXECUTOR_JAN_11_2013_0831PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        class thread_pool_executor
          : public threads::detail::executor_base
        {
        public:
            thread_pool_executor(std::size_t num_threads = 1);
            ~thread_pool_executor();

            /// Schedule the specified function for execution in this executor.
            /// Depending on the subclass implementation, this may block in some
            /// situations.
            void add(HPX_STD_FUNCTION<void()> f, char const* description);

            /// Like add(), except that if the attempt to add the function would
            /// cause the caller to block in add, try_add would instead do
            /// nothing and return false.
            bool try_add(HPX_STD_FUNCTION<void()> f, char const* description);

            /// Schedule given function for execution in this executor no sooner
            /// than time abs_time. This call never blocks, and may violate
            /// bounds on the executor's queue size.
            void add_at(boost::posix_time::ptime const& abs_time,
                HPX_STD_FUNCTION<void()> f, char const* description);

            /// Schedule given function for execution in this executor no sooner
            /// than time rel_time from now. This call never blocks, and may
            /// violate bounds on the executor's queue size.
            void add_after(boost::posix_time::time_duration const& rel_time,
                HPX_STD_FUNCTION<void()> f, char const* description);

            // Return an estimate of the number of waiting tasks.
            std::size_t num_pending_tasks() const;

        private:
            // internal run method
            void run(std::size_t num_thread);

            // the scheduler used by this executor
            policies::local_queue_scheduler<lcos::local::spinlock> scheduler_;
            lcos::local::counting_semaphore shutdown_sem_;
            boost::atomic<hpx::state> state_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct serial_executor : public executor
    {
        serial_executor(std::size_t num_queues = 1)
          : executor(new detail::thread_pool_executor(num_queues))
        {}
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

