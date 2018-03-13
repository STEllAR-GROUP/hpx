//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace hpx { namespace threads { namespace executors
{
    namespace detail
    {
        pool_executor::pool_executor(std::string const& pool_name)
          : pool_(hpx::threads::get_thread_manager().get_pool(pool_name))
          , stacksize_(thread_stacksize_default)
          , priority_(thread_priority_default)
        {}

        pool_executor::pool_executor(std::string const& pool_name,
            thread_stacksize stacksize)
          : pool_(hpx::threads::get_thread_manager().get_pool(pool_name))
          , stacksize_(stacksize)
          , priority_(thread_priority_default)
        {}

        pool_executor::pool_executor(std::string const& pool_name,
            thread_priority priority, thread_stacksize stacksize)
          : pool_(hpx::threads::get_thread_manager().get_pool(pool_name))
          , stacksize_(stacksize)
          , priority_(priority)
        {}

        threads::thread_result_type
        pool_executor::thread_function_nullary(closure_type func)
        {
            // execute the actual thread function
            func();
            return threads::thread_result_type(threads::terminated,
                threads::invalid_thread_id);
        }

        // Return the requested policy element
        std::size_t pool_executor::get_policy_element(
            threads::detail::executor_parameter p, error_code& ec) const
        {
            //! FIXME what is this supposed to do??

            HPX_THROWS_IF(ec, bad_parameter,
                "pool_executor::get_policy_element",
                "requested value of invalid policy element");
            return std::size_t(-1);
        }

        // Schedule the specified function for execution in this executor.
        // Depending on the subclass implementation, this may block in some
        // situations.
        void pool_executor::add(closure_type&& f,
            util::thread_description const& desc,
            threads::thread_state_enum initial_state,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            // create a new thread
            thread_init_data data(
                util::bind(
                    util::one_shot(
                        &pool_executor::thread_function_nullary),
                    std::move(f)),
                desc);

            if (stacksize == threads::thread_stacksize_default)
                stacksize = stacksize_;
            data.stacksize = threads::get_stack_size(stacksize);
            data.priority = priority_;

            threads::thread_id_type id = threads::invalid_thread_id;
            pool_.create_thread(data, id, initial_state, ec);
            if (ec)
                return;

            HPX_ASSERT(invalid_thread_id != id);

            if (&ec != &throws)
                ec = make_success_code();
        }

        // Schedule given function for execution in this executor no sooner
        // than time abs_time. This call never blocks, and may violate
        // bounds on the executor's queue size.
        void pool_executor::add_at(
            util::steady_clock::time_point const& abs_time,
            closure_type&& f, util::thread_description const& desc,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            // create a new suspended thread
            thread_init_data data(
                util::bind(
                    util::one_shot(
                        &pool_executor::thread_function_nullary),
                    std::move(f)),
                desc);

            if (stacksize == threads::thread_stacksize_default)
                stacksize = stacksize_;
            data.stacksize = threads::get_stack_size(stacksize);
            data.priority = priority_;

            threads::thread_id_type id = threads::invalid_thread_id;
            pool_.create_thread(data, id, suspended, ec);
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
        void pool_executor::add_after(
            util::steady_clock::duration const& rel_time, closure_type&& f,
            util::thread_description const& desc,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            return add_at(util::steady_clock::now() + rel_time,
                std::move(f), desc, stacksize, ec);
        }

        // Return an estimate of the number of waiting tasks.
        std::uint64_t pool_executor::num_pending_closures(
            error_code& ec) const
        {
            if (&ec != &throws)
                ec = make_success_code();

            std::lock_guard<mutex_type> lk(mtx_);
            return pool_.get_thread_count(
                unknown, thread_priority_default, std::size_t(-1), false);
        }

        // Reset internal (round robin) thread distribution scheme
        void pool_executor::reset_thread_distribution()
        {
            pool_.reset_thread_distribution();
        }
    }
}}}

namespace hpx { namespace threads { namespace executors
{
    pool_executor::pool_executor(std::string const& pool_name)
        : scheduled_executor(new detail::pool_executor(pool_name))
    {
    }

    pool_executor::pool_executor(std::string const& pool_name,
            thread_stacksize stacksize)
      : scheduled_executor(new detail::pool_executor(pool_name, stacksize))
    {
    }

    pool_executor::pool_executor(std::string const& pool_name,
            thread_priority priority, thread_stacksize stacksize)
      : scheduled_executor(
            new detail::pool_executor(pool_name, priority, stacksize))
    {
    }
}}}
