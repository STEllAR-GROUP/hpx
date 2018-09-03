//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/executors/service_executors.hpp>

#include <hpx/config/asio.hpp>
#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/util/unique_function.hpp>

#include <boost/asio/basic_waitable_timer.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <iostream>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    service_executor::service_executor(
            char const* pool_name, char const* pool_name_suffix)
      : pool_(get_thread_pool(pool_name, pool_name_suffix)),
        task_count_(0),
        blocking_(true)
    {
        if (!pool_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "service_executor::service_executor",
                "couldn't retrieve thread pool: " + std::string(pool_name));
        }
    }

    service_executor::~service_executor()
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (blocking_)
        {
            while (task_count_ > 0)
            {
                // We need to cancel the wait process here, since we might block
                // other running HPX threads.
                shutdown_cv_.wait_for(l, std::chrono::seconds(1));
                if (hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend();
                }
            }
        }
    }

    void service_executor::detach()
    {
        std::unique_lock<mutex_type> l(mtx_);
        blocking_ = false;
    }

    void service_executor::thread_wrapper(closure_type&& f) //-V669
    {
        f();                          // execute the actual thread function

        // By hanging on to the lock during notify_all, we ensure that the
        // destructor is only completed after this function returned
        {
            std::unique_lock<mutex_type> l(mtx_);
            HPX_ASSERT(task_count_ > 0);
            if (--task_count_ == 0)
            {
                shutdown_cv_.notify_all();
            }
        }
    }

    struct thread_wrapper_helper
    {
        typedef void result_type;

        thread_wrapper_helper(
                service_executor* exec, service_executor::closure_type&& f)
          : exec_(exec)
          , f_(std::move(f))
        {
            intrusive_ptr_add_ref(exec);
        }

        result_type invoke()
        {
            exec_->thread_wrapper(std::move(f_));
            intrusive_ptr_release(exec_);
        }

        service_executor* exec_;
        service_executor::closure_type f_;
    };

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void service_executor::add(closure_type&& f,
        util::thread_description const& desc,
        threads::thread_state_enum initial_state, bool run_now,
        threads::thread_stacksize stacksize,
        threads::thread_schedule_hint schedulehint,
        error_code& ec)
    {
        ++task_count_;

        std::shared_ptr<thread_wrapper_helper> wfp(
            std::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(
            util::bind(&thread_wrapper_helper::invoke, wfp));
    }

    void service_executor::add_no_count(closure_type&& f)
    {
        std::shared_ptr<thread_wrapper_helper> wfp(
            std::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(
            util::bind(&thread_wrapper_helper::invoke, wfp));
    }

    typedef boost::asio::basic_waitable_timer<util::steady_clock> deadline_timer;

    struct delayed_add_helper
    {
        typedef void result_type;

        delayed_add_helper(service_executor* exec,
                service_executor::closure_type&& f,
                boost::asio::io_service& io_service,
                util::steady_clock::time_point const& abs_time)
          : exec_(exec)
          , f_(std::move(f))
          , timer_(io_service, abs_time)
        {
            intrusive_ptr_add_ref(exec);
        }

        result_type invoke()
        {
            exec_->add_no_count(std::move(f_));
            intrusive_ptr_release(exec_);
        }

        service_executor* exec_;
        service_executor::closure_type f_;
        deadline_timer timer_;
    };

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        util::steady_clock::time_point const& abs_time,
        closure_type&& f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        std::shared_ptr<delayed_add_helper> wfp(
            std::make_shared<delayed_add_helper>(
                this, std::move(f), pool_->get_io_service(), abs_time));

        wfp->timer_.async_wait(
            util::bind(&delayed_add_helper::invoke, wfp));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        util::steady_clock::duration const& rel_time,
        closure_type&& f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(util::steady_clock::now() + rel_time,
            std::move(f), desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    std::uint64_t service_executor::num_pending_closures(error_code& ec) const
    {
        return task_count_;
    }

    // Return the requested policy element
    std::size_t service_executor::get_policy_element(
        threads::detail::executor_parameter p, error_code& ec) const
    {
        switch(p) {
        case threads::detail::min_concurrency:
        case threads::detail::max_concurrency:
        case threads::detail::current_concurrency:
            return pool_->size();

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }
}}}}
