//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/executors/service_executors.hpp>
#include <hpx/util/bind.hpp>

#include <boost/asio/basic_deadline_timer.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    service_executor::service_executor(
            char const* pool_name, char const* pool_name_suffix)
      : pool_(get_thread_pool(pool_name, pool_name_suffix)),
        task_count_(0), shutdown_sem_(0)
    {
        if (!pool_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "service_executor::service_executor",
                "couldn't retrieve thread pool: " + std::string(pool_name));
        }
    }

    service_executor::~service_executor()
    {
        if (task_count_ != 0)
            shutdown_sem_.wait();
    }

    void service_executor::thread_wrapper(closure_type && f) //-V669
    {
        f();                          // execute the actual thread function

        if (--task_count_ == 0)
            shutdown_sem_.signal();
    }

    struct thread_wrapper_helper
    {
        typedef void result_type;

        thread_wrapper_helper(
            service_executor* exec
          , service_executor::closure_type && f
        ) : exec_(exec)
          , f_(std::move(f))
        {}

        result_type invoke()
        {
            exec_->thread_wrapper(std::move(f_));
        }

        service_executor* exec_;
        service_executor::closure_type f_;
    };

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void service_executor::add(closure_type && f,
        util::thread_description const& desc,
        threads::thread_state_enum initial_state, bool run_now,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<thread_wrapper_helper> wfp(
            boost::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(
            util::bind(&thread_wrapper_helper::invoke, wfp));
    }

    void service_executor::add_no_count(closure_type && f)
    {
        boost::shared_ptr<thread_wrapper_helper> wfp(
            boost::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(
            util::bind(&thread_wrapper_helper::invoke, wfp));
    }

    typedef boost::asio::basic_deadline_timer<
        boost::chrono::steady_clock
        , util::chrono_traits<boost::chrono::steady_clock>
    > steady_clock_deadline_timer;

    struct delayed_add_helper
    {
        typedef void result_type;

        delayed_add_helper(
            service_executor* exec
          , service_executor::closure_type && f
          , boost::asio::io_service& io_service
          , boost::chrono::steady_clock::time_point const& abs_time
        ) : exec_(exec)
          , f_(std::move(f))
          , timer_(io_service, abs_time)
        {}

        result_type invoke()
        {
            exec_->add_no_count(std::move(f_));
        }

        service_executor* exec_;
        service_executor::closure_type f_;
        steady_clock_deadline_timer timer_;
    };

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        boost::chrono::steady_clock::time_point const& abs_time,
        closure_type && f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<delayed_add_helper> wfp(
            boost::make_shared<delayed_add_helper>(
                this, std::move(f), pool_->get_io_service(), abs_time));

        wfp->timer_.async_wait(
            util::bind(&delayed_add_helper::invoke, wfp));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        boost::chrono::steady_clock::duration const& rel_time,
        closure_type && f, util::thread_description const& desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        return add_at(boost::chrono::steady_clock::now() + rel_time,
            std::move(f), desc, stacksize, ec);
    }

    // Return an estimate of the number of waiting tasks.
    boost::uint64_t service_executor::num_pending_closures(error_code& ec) const
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
