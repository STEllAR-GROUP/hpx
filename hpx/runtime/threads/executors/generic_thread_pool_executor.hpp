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
        class HPX_EXPORT generic_thread_pool_executor
          : public threads::detail::executor_base
        {
        public:
            generic_thread_pool_executor(policies::scheduler_base* scheduler);
            ~generic_thread_pool_executor();

            // Schedule the specified function for execution in this executor.
            // Depending on the subclass implementation, this may block in some
            // situations.
            void add(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* description,
                threads::thread_state_enum initial_state, bool run_now,
                error_code& ec);

            // Return an estimate of the number of waiting tasks.
            std::size_t num_pending_closures(error_code& ec) const;

        protected:
            static threads::thread_state_enum thread_function_nullary(
                HPX_STD_FUNCTION<void()> const& func);

        private:
            policies::scheduler_base* scheduler_base_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT generic_thread_pool_executor : public executor
    {
        generic_thread_pool_executor(policies::scheduler_base* scheduler);
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif

