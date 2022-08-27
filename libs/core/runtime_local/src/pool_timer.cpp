//  Copyright (c) 2016 Bibek Wagle
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/pool_timer.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <asio/basic_waitable_timer.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace hpx { namespace util { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    class pool_timer : public std::enable_shared_from_this<pool_timer>
    {
    private:
        friend class util::pool_timer;

        using mutex_type = lcos::local::spinlock;

    public:
        pool_timer();

        pool_timer(hpx::function<bool()> const& f,
            hpx::function<void()> const& on_term,
            std::string const& description, bool pre_shutdown);

        ~pool_timer();

        bool start(
            hpx::chrono::steady_duration const& time_duration, bool evaluate);
        bool stop();

        bool is_started() const
        {
            return is_started_;
        }
        bool is_terminated() const
        {
            return is_terminated_;
        }
        void timer_handler(const std::error_code&);

        void terminate();    // handle system shutdown
        bool stop_locked();

    private:
        using deadline_timer =
            asio::basic_waitable_timer<std::chrono::steady_clock>;

        mutable mutex_type mtx_;
        hpx::function<bool()> f_;          ///< function to call
        hpx::function<void()> on_term_;    ///< function to call on termination
        std::string description_;    ///< description of this interval timer

        bool pre_shutdown_;     ///< execute termination during pre-shutdown
        bool is_started_;       ///< timer has been started (is running)
        bool first_start_;      ///< flag to distinguish first start invocation
        bool is_terminated_;    ///< The timer has been terminated
        bool is_stopped_;
        deadline_timer* timer_;
    };

    ///////////////////////////////////////////////////////////////////////////
    pool_timer::pool_timer()
      : pre_shutdown_()
      , is_started_(false)
      , first_start_(true)
      , is_terminated_(false)
      , is_stopped_(false)
      , timer_(new deadline_timer(
            hpx::get_runtime().get_thread_pool("timer_pool")->get_io_service()))
    {
    }

    pool_timer::pool_timer(hpx::function<bool()> const& f,
        hpx::function<void()> const& on_term, std::string const& description,
        bool pre_shutdown)
      : f_(f)
      , on_term_(on_term)
      , description_(description)
      , pre_shutdown_(pre_shutdown)
      , is_started_(false)
      , first_start_(true)
      , is_terminated_(false)
      , is_stopped_(false)
      , timer_(new deadline_timer(
            hpx::get_runtime().get_thread_pool("timer_pool")->get_io_service()))
    {
    }

    void pool_timer::timer_handler(const std::error_code& err)
    {
        if (!is_stopped_ || !is_terminated_)
        {
            is_started_ = false;
            if (!err)
                f_();
        }
    }

    bool pool_timer::start(
        hpx::chrono::steady_duration const& time_duration, bool /* evaluate_ */)
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (is_terminated_)
            return false;

        if (!is_started_)
        {
            is_stopped_ = false;
            is_started_ = true;

            if (first_start_)
            {
                first_start_ = false;

                util::unlock_guard<std::unique_lock<mutex_type>> ul(l);
                if (pre_shutdown_)
                {
                    register_pre_shutdown_function(util::deferred_call(
                        &pool_timer::terminate, this->shared_from_this()));
                }
                else
                {
                    register_shutdown_function(util::deferred_call(
                        &pool_timer::terminate, this->shared_from_this()));
                }
            }

            HPX_ASSERT(timer_ != nullptr);
            timer_->expires_from_now(time_duration.value());
            timer_->async_wait(hpx::bind_front(
                &pool_timer::timer_handler, this->shared_from_this()));

            return true;
        }
        return false;
    }

    bool pool_timer::stop()
    {
        std::lock_guard<mutex_type> l(mtx_);
        return stop_locked();
    }

    bool pool_timer::stop_locked()
    {
        if (!is_terminated_ && is_started_ && !is_stopped_)
        {
            is_started_ = false;
            is_stopped_ = true;

            HPX_ASSERT(timer_ != nullptr);
            timer_->cancel();
            return true;
        }
        return false;
    }

    void pool_timer::terminate()
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (!is_terminated_)
        {
            is_terminated_ = true;
            stop_locked();

            if (on_term_)
            {
                l.unlock();
                on_term_();
            }
        }
        delete timer_;
        timer_ = nullptr;
    }

    pool_timer::~pool_timer()
    {
        try
        {
            terminate();
        }
        catch (...)
        {
            ;    // there is nothing we can do here
        }
    }
}}}    // namespace hpx::util::detail

namespace hpx { namespace util {
    pool_timer::pool_timer() {}

    pool_timer::pool_timer(hpx::function<bool()> const& f,
        hpx::function<void()> const& on_term, std::string const& description,
        bool pre_shutdown)
      : timer_(std::make_shared<detail::pool_timer>(
            f, on_term, description, pre_shutdown))
    {
    }

    pool_timer::~pool_timer()
    {
        timer_->terminate();
    }

    bool pool_timer::start(
        hpx::chrono::steady_duration const& time_duration, bool evaluate)
    {
        return timer_->start(time_duration, evaluate);
    }

    bool pool_timer::stop()
    {
        return timer_->stop();
    }

    bool pool_timer::is_started() const
    {
        return timer_->is_started();
    }

    bool pool_timer::is_terminated() const
    {
        return timer_->is_terminated();
    }
}}    // namespace hpx::util
