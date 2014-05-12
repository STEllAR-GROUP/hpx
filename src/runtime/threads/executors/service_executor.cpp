//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/executors/service_executor.hpp>
#include <hpx/util/bind.hpp>

#include <boost/asio/deadline_timer.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    service_executor::service_executor(char const* pool_name)
      : pool_(get_thread_pool(pool_name)), task_count_(0), shutdown_sem_(0)
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

    void service_executor::thread_wrapper(closure_type && f)
    {
        f();                          // execute the actual thread function

        if (--task_count_ == 0)
            shutdown_sem_.signal();
    }

#if defined(BOOST_ASIO_HAS_MOVE)
    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void service_executor::add(closure_type && f,
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        pool_->get_io_service().post(util::bind(
            util::one_shot(&service_executor::thread_wrapper),
            this, std::move(f)));
    }

    void service_executor::add_no_count(closure_type && f)
    {
        pool_->get_io_service().post(util::bind(
            util::one_shot(&service_executor::thread_wrapper),
            this, std::move(f)));
    }
#else
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
        char const* desc, threads::thread_state_enum initial_state,
        bool run_now, threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<thread_wrapper_helper> wfp(
            boost::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(util::bind(
            util::one_shot(&thread_wrapper_helper::invoke), wfp));
    }

    void service_executor::add_no_count(closure_type && f)
    {
        boost::shared_ptr<thread_wrapper_helper> wfp(
            boost::make_shared<thread_wrapper_helper>(
                this, std::move(f)));

        pool_->get_io_service().post(util::bind(
            util::one_shot(&thread_wrapper_helper::invoke), wfp));
    }
#endif

#if defined(BOOST_ASIO_HAS_MOVE)
    static void delayed_add(
        boost::intrusive_ptr<service_executor> this_,
        service_executor::closure_type && f,
        boost::shared_ptr<boost::asio::deadline_timer>)
    {
        this_->add_no_count(std::move(f));
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), abs_time));

        t->async_wait(util::bind(
            util::one_shot(&delayed_add),
            this, std::move(f), t));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), rel_time));

        t->async_wait(util::bind(
            util::one_shot(&delayed_add),
            this, std::move(f), t));
    }
#else
    struct delayed_add_helper
    {
        typedef void result_type;

        delayed_add_helper(
            service_executor* exec
          , service_executor::closure_type && f
          , boost::asio::io_service& io_service
          , boost::posix_time::ptime const& abs_time
        ) : exec_(exec)
          , f_(std::move(f))
          , timer_(io_service, abs_time)
        {}

        delayed_add_helper(
            service_executor* exec
          , service_executor::closure_type && f
          , boost::asio::io_service& io_service
          , boost::posix_time::time_duration const& rel_time
        ) : exec_(exec)
          , f_(std::move(f))
          , timer_(io_service, rel_time)
        {}

        result_type invoke()
        {
            exec_->add_no_count(std::move(f_));
        }

        service_executor* exec_;
        service_executor::closure_type f_;
        boost::asio::deadline_timer timer_;
    };

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<delayed_add_helper> wfp(
            boost::make_shared<delayed_add_helper>(
                this, std::move(f), pool_->get_io_service(), abs_time));

        wfp->timer_.async_wait(util::bind(
            util::one_shot(&delayed_add_helper::invoke), wfp));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        closure_type && f, char const* desc,
        threads::thread_stacksize stacksize, error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<delayed_add_helper> wfp(
            boost::make_shared<delayed_add_helper>(
                this, std::move(f), pool_->get_io_service(), rel_time));

        wfp->timer_.async_wait(util::bind(
            util::one_shot(&delayed_add_helper::invoke), wfp));
    }
#endif

    // Return an estimate of the number of waiting tasks.
    std::size_t service_executor::num_pending_closures(error_code& ec) const
    {
        return task_count_;
    }
}}}}
