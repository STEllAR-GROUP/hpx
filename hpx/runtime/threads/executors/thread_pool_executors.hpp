//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_THREADS_EXECUTORS_SERIAL_EXECUTOR_JAN_11_2013_0831PM)
#define HPX_RUNTIME_THREADS_EXECUTORS_SERIAL_EXECUTOR_JAN_11_2013_0831PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
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
        template <typename Scheduler>
        class manage_thread_pool_executor;

        //////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class HPX_EXPORT thread_pool_executor
          : public threads::detail::executor_base
        {
        public:
            thread_pool_executor(std::size_t max_punits = 1,
                std::size_t min_punits = 1);
            ~thread_pool_executor();

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

        protected:
            friend class manage_thread_pool_executor<Scheduler>;

            // The function below are used by the resource manager to
            // interact with the scheduler.

            // Return the requested policy element
            std::size_t get_policy_element(threads::detail::executor_parameter p,
                error_code& ec) const;

            // Return statistics collected by this scheduler
            void get_statistics(executor_statistics& stats, error_code& ec) const;

            // Provide the given processing unit to the scheduler.
            void add_processing_unit(std::size_t virt_core,
                std::size_t thread_num, error_code& ec);

            // Remove the given processing unit from the scheduler.
            void remove_processing_unit(std::size_t thread_num, error_code& ec);

        private:
            // internal run method
            void run(std::size_t num_thread);

            threads::thread_state_enum thread_function_nullary(
                HPX_STD_FUNCTION<void()> const& func);

            // the scheduler used by this executor
            Scheduler scheduler_;
            lcos::local::counting_semaphore shutdown_sem_;
            boost::ptr_vector<boost::atomic<hpx::state> > states_;

            // map internal virtual core number to punit numbers
            std::vector<std::size_t> puinits_;

            // collect statistics
            boost::atomic<std::size_t> current_concurrency_;
            boost::atomic<boost::uint64_t> tasks_scheduled_;
            boost::atomic<boost::uint64_t> tasks_completed_;

            // policy elements
            std::size_t const max_punits_;
            std::size_t const min_punits_;

            // resource manager registration
            std::size_t cookie_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT local_queue_executor : public executor
    {
        local_queue_executor(std::size_t max_punits = 1, 
            std::size_t min_punits = 1);
    };

    struct HPX_EXPORT local_priority_queue_executor : public executor
    {
        local_priority_queue_executor(std::size_t max_punits = 1, 
            std::size_t min_punits = 1);
    };

#if defined(HPX_STATIC_PRIORITY_SCHEDULER)
    struct HPX_EXPORT static_priority_queue_executor : public executor
    {
        static_priority_queue_executor(std::size_t max_punits = 1, 
            std::size_t min_punits = 1);
    };
#endif
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

