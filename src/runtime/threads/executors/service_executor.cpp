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
      : pool_(get_thread_pool(pool_name))
    {
        if (!pool_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "service_executor::service_executor",
                "couldn't retrieve thread pool: " + std::string(pool_name));
        }
    }

    service_executor::~service_executor()
    {
    }

    // Schedule the specified function for execution in this executor.
    // Depending on the subclass implementation, this may block in some
    // situations.
    void service_executor::add(HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        pool_->get_io_service().post(boost::move(f));
    }

    // Like add(), except that if the attempt to add the function would
    // cause the caller to block in add, try_add would instead do
    // nothing and return false.
    bool service_executor::try_add(HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        pool_->get_io_service().post(boost::move(f));
        return true;      // this function will never block
    }

    static void delayed_add(service_executor* this_,
        HPX_STD_FUNCTION<void()> f, char const* desc,
        boost::shared_ptr<boost::asio::deadline_timer> t)
    {
        this_->add(f, desc);
    }

    // Schedule given function for execution in this executor no sooner
    // than time abs_time. This call never blocks, and may violate
    // bounds on the executor's queue size.
    void service_executor::add_at(
        boost::posix_time::ptime const& abs_time,
        HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), abs_time));
        t->async_wait(util::bind(&delayed_add, this, f, desc, t));
    }

    // Schedule given function for execution in this executor no sooner
    // than time rel_time from now. This call never blocks, and may
    // violate bounds on the executor's queue size.
    void service_executor::add_after(
        boost::posix_time::time_duration const& rel_time,
        HPX_STD_FUNCTION<void()> f, char const* desc)
    {
        boost::shared_ptr<boost::asio::deadline_timer> t(
            boost::make_shared<boost::asio::deadline_timer>(
                pool_->get_io_service(), rel_time));
        t->async_wait(util::bind(&delayed_add, this, f, desc, t));
    }

    // Return an estimate of the number of waiting tasks.
    std::size_t service_executor::num_pending_tasks() const
    {
        return std::size_t(-1);     // we do not support this functionality
    }
}}}}
