//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/runtime/threads/executors/customized_pool_executors.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>

#include <string>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        customized_pool_executor::customized_pool_executor(
                std::string const& pool_name)
          : pool_(hpx::threads::get_thread_manager().get_pool(pool_name))
          , scheduler_(pool_.get_scheduler())
        {
            //! FIXME
            //! throw exception if name = default: tell them to use default_executor
            //! throw exception if above = nullptr
        }

        threads::thread_result_type
        customized_pool_executor::thread_function_nullary(closure_type func)
        {
            // execute the actual thread function
            func();
            return threads::thread_result_type(threads::terminated, nullptr);
        }

        // Return the requested policy element
        std::size_t customized_pool_executor::get_policy_element(
            threads::detail::executor_parameter p, error_code& ec) const
        {
            //! FIXME what is this supposed to do??

            HPX_THROWS_IF(ec, bad_parameter,
                "customized_pool_executor::get_policy_element",
                "requested value of invalid policy element");
            return std::size_t(-1);
        }

        // Schedule the specified function for execution in this executor.
        // Depending on the subclass implementation, this may block in some
        // situations.
        void customized_pool_executor::add(closure_type&& f,
            util::thread_description const& desc,
            threads::thread_state_enum initial_state, bool run_now,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            // create a new thread
            thread_init_data data(
                util::bind(
                    util::one_shot(
                        &customized_pool_executor::thread_function_nullary),
                    std::move(f)),
                desc);
            data.stacksize = threads::get_stack_size(stacksize);

            threads::thread_id_type id = threads::invalid_thread_id;
            pool_.create_thread(data, id, initial_state, run_now, ec);
            if (ec)
                return;

            HPX_ASSERT(invalid_thread_id != id || !run_now);

            if (&ec != &throws)
                ec = make_success_code();
        }

        // Schedule given function for execution in this executor no sooner
        // than time abs_time. This call never blocks, and may violate
        // bounds on the executor's queue size.
        void customized_pool_executor::add_at(
            util::steady_clock::time_point const& abs_time,
            closure_type&& f, util::thread_description const& desc,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            // create a new suspended thread
            thread_init_data data(
                util::bind(
                    util::one_shot(
                        &customized_pool_executor::thread_function_nullary),
                    std::move(f)),
                desc);
            data.stacksize = threads::get_stack_size(stacksize);

            threads::thread_id_type id = threads::invalid_thread_id;
            pool_.create_thread(data, id, suspended, true, ec);
            if (ec)
                return;

            HPX_ASSERT(invalid_thread_id != id);    // would throw otherwise

            // now schedule new thread for execution
            pool_.set_state(abs_time, id, pending, wait_timeout,
                thread_priority_normal, ec);
            if (ec)
                return;

            if (&ec != &throws)
                ec = make_success_code();
        }

        // Schedule given function for execution in this executor no sooner
        // than time rel_time from now. This call never blocks, and may
        // violate bounds on the executor's queue size.
        void customized_pool_executor::add_after(
            util::steady_clock::duration const& rel_time, closure_type&& f,
            util::thread_description const& desc,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            return add_at(util::steady_clock::now() + rel_time,
                std::move(f), desc, stacksize, ec);
        }

        // Return an estimate of the number of waiting tasks.
        std::uint64_t customized_pool_executor::num_pending_closures(
            error_code& ec) const
        {
            if (&ec != &throws)
                ec = make_success_code();

            std::lock_guard<mutex_type> lk(mtx_);
            return pool_.get_thread_count(
                unknown, thread_priority_default, std::size_t(-1), false);
        }

        // Reset internal (round robin) thread distribution scheme
        void customized_pool_executor::reset_thread_distribution()
        {
            pool_.reset_thread_distribution();
        }
    }
}}}

namespace hpx { namespace threads { namespace executors
{
    customized_pool_executor::customized_pool_executor(
        const std::string& pool_name)
      : scheduled_executor(new detail::customized_pool_executor(pool_name))
    {
    }
}}}
