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

    void service_executor::thread_wrapper(HPX_STD_FUNCTION<void()> const& f)
    {
        f();                          // execute the actual thread function

        if (--task_count_ == 0)
            shutdown_sem_.signal();
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void service_executor::add(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, 
        char const* desc, threads::thread_state_enum initial_state, 
        bool run_now, error_code& ec)
    {
        ++task_count_;

        pool_->get_io_service().post(util::bind(
            &service_executor::thread_wrapper, this, boost::move(f)));
    }

    void service_executor::add_no_count(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f)
    {
        pool_->get_io_service().post(util::bind(
            &service_executor::thread_wrapper, this, boost::move(f)));
    }

    static void delayed_add(
        boost::intrusive_ptr<service_executor> this_,
        HPX_STD_FUNCTION<void()> f,
        boost::shared_ptr<boost::asio::deadline_timer>)
    {
        this_->add_no_count(boost::move(f));
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* desc, 
        error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), abs_time));

        t->async_wait(util::bind(&delayed_add, this, boost::move(f), t));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        BOOST_RV_REF(HPX_STD_FUNCTION<void()>) f, char const* desc, 
        error_code& ec)
    {
        ++task_count_;

        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), rel_time));

        t->async_wait(util::bind(&delayed_add, this, boost::move(f), t));
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t service_executor::num_pending_closures(error_code& ec) const
    {
        return task_count_;
    }
}}}}
