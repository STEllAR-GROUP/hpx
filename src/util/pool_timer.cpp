//  Copyright (c) 2016 Bibek Wagle
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/pool_timer.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/util/bind.hpp>

#include <boost/thread/locks.hpp>
#include <boost/make_shared.hpp>

#include <hpx/include/runtime.hpp>
#include <hpx/util/io_service_pool.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    pool_timer::pool_timer()
        : mytimer_(hpx::get_runtime().get_thread_pool("timer_pool")->get_io_service())
    {}

    pool_timer::pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::posix_time::time_duration microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(on_term),
        microsecs_(microsecs), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false), is_stopped_(false),
        mytimer_(hpx::get_runtime().get_thread_pool("timer_pool")->get_io_service())
    {}

    void pool_timer::dummy(const boost::system::error_code &)
    {
        f_();
    }

    bool pool_timer::start(bool evaluate_)
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (is_terminated_)
            return false;

        if (!is_started_) {
            if (first_start_) {
                first_start_ = false;

                util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                if (pre_shutdown_)
                {
                    register_pre_shutdown_function(
                        util::bind(&pool_timer::terminate,
                            this->shared_from_this()));
                }
                else
                {
                    register_shutdown_function(
                        util::bind(&pool_timer::terminate,
                            this->shared_from_this()));
                }
            }

            is_stopped_ = false;
            is_started_ = true;
            mytimer_.expires_from_now(microsecs_);
            boost::system::error_code e;
            mytimer_.async_wait( util::bind(&pool_timer::dummy,
                this->shared_from_this(),e));
            l.unlock();
            return true;
        }
        return false;
    }

    bool pool_timer::stop()
    {
        boost::lock_guard<mutex_type> l(mtx_);
        is_stopped_ = true;
        return stop_locked();
    }

    bool pool_timer::stop_locked()
    {
        if (is_started_) {
            is_started_ = false;
            mytimer_.cancel();
            return true;
        }
        return false;
    }

    void pool_timer::terminate()
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (!is_terminated_) {
            is_terminated_ = true;
            stop_locked();

            if (on_term_) {
                l.unlock();
                on_term_();
            }
        }
    }

    pool_timer::~pool_timer()
    {
        try {
            terminate();
        }
        catch(...) {
            ;   // there is nothing we can do here
        }
    }

}}}

namespace hpx { namespace util
{
    pool_timer::pool_timer() {}

    pool_timer::pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::posix_time::time_duration microsecs, std::string const& description,
            bool pre_shutdown)
      : timer_(boost::make_shared<detail::pool_timer>(
            f, on_term, microsecs, description, pre_shutdown))
    {}

    pool_timer::~pool_timer()
    {
        timer_->terminate();
    }
}}
