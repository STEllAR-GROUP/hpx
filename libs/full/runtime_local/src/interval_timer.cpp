//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_local/interval_timer.hpp>
#include <hpx/runtime_local/shutdown_function.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    interval_timer::interval_timer()
      : microsecs_(0)
      , id_()
      , timerid_()
      , pre_shutdown_(false)
      , is_started_(false)
      , first_start_(true)
      , is_terminated_(false)
      , is_stopped_(false)
    {
    }

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
        std::int64_t microsecs, std::string const& description,
        bool pre_shutdown)
      : f_(f)
      , on_term_()
      , microsecs_(microsecs)
      , id_()
      , timerid_()
      , description_(description)
      , pre_shutdown_(pre_shutdown)
      , is_started_(false)
      , first_start_(true)
      , is_terminated_(false)
      , is_stopped_(false)
    {
    }

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
        util::function_nonser<void()> const& on_term, std::int64_t microsecs,
        std::string const& description, bool pre_shutdown)
      : f_(f)
      , on_term_(on_term)
      , microsecs_(microsecs)
      , id_()
      , timerid_()
      , description_(description)
      , pre_shutdown_(pre_shutdown)
      , is_started_(false)
      , first_start_(true)
      , is_terminated_(false)
      , is_stopped_(false)
    {
    }

    bool interval_timer::start(bool evaluate_)
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (is_terminated_)
            return false;

        if (!is_started_)
        {
            if (first_start_)
            {
                first_start_ = false;

                util::unlock_guard<std::unique_lock<mutex_type>> ul(l);
                if (pre_shutdown_)
                {
                    register_pre_shutdown_function(util::deferred_call(
                        &interval_timer::terminate, this->shared_from_this()));
                }
                else
                {
                    register_shutdown_function(util::deferred_call(
                        &interval_timer::terminate, this->shared_from_this()));
                }
            }

            is_stopped_ = false;

            if (evaluate_)
            {
                l.unlock();
                evaluate(threads::thread_restart_state::signaled);
            }
            else
            {
                schedule_thread(l);
            }

            return true;
        }
        return false;
    }

    bool interval_timer::restart(bool evaluate_)
    {
        if (!is_started_)
            return start(evaluate_);

        std::unique_lock<mutex_type> l(mtx_);

        if (is_terminated_)
            return false;

        // interrupt timer thread, if needed
        stop_locked();

        // reschedule evaluation thread
        if (evaluate_)
        {
            l.unlock();
            evaluate(threads::thread_restart_state::signaled);
        }
        else
        {
            schedule_thread(l);
        }
        return true;
    }

    bool interval_timer::stop(bool terminate_timer)
    {
        if (terminate_timer)
        {
            terminate();
            return true;
        }

        std::lock_guard<mutex_type> l(mtx_);
        is_stopped_ = true;
        return stop_locked();
    }

    bool interval_timer::stop_locked()
    {
        if (is_started_)
        {
            is_started_ = false;

            if (timerid_)
            {
                error_code ec(lightweight);    // avoid throwing on error
                threads::set_thread_state(timerid_,
                    threads::thread_schedule_state::pending,
                    threads::thread_restart_state::abort,
                    threads::thread_priority::boost, true, ec);
                timerid_.reset();
            }
            if (id_)
            {
                error_code ec(lightweight);    // avoid throwing on error
                threads::set_thread_state(id_,
                    threads::thread_schedule_state::pending,
                    threads::thread_restart_state::abort,
                    threads::thread_priority::boost, true, ec);
                id_.reset();
            }
            return true;
        }

        HPX_ASSERT(id_ == nullptr);
        HPX_ASSERT(timerid_ == nullptr);
        return false;
    }

    void interval_timer::terminate()
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
    }

    interval_timer::~interval_timer()
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

    std::int64_t interval_timer::get_interval() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return microsecs_;
    }

    void interval_timer::change_interval(std::int64_t new_interval)
    {
        HPX_ASSERT(new_interval > 0);

        std::lock_guard<mutex_type> l(mtx_);
        microsecs_ = new_interval;
    }

    threads::thread_result_type interval_timer::evaluate(
        threads::thread_restart_state statex)
    {
        try
        {
            std::unique_lock<mutex_type> l(mtx_);

            if (is_stopped_ || is_terminated_ ||
                statex == threads::thread_restart_state::abort ||
                0 == microsecs_)
            {
                // object has been finalized, exit
                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }

            if (id_ != nullptr && id_ != threads::get_self_id())
            {
                // obsolete timer thread
                return threads::thread_result_type(
                    threads::thread_schedule_state::terminated,
                    threads::invalid_thread_id);
            }

            id_.reset();
            timerid_.reset();
            is_started_ = false;

            bool result = false;

            {
                util::unlock_guard<std::unique_lock<mutex_type>> ul(l);
                result = f_();    // invoke the supplied function
            }

            // some other thread might already have started the timer
            if (nullptr == id_ && result)
            {
                HPX_ASSERT(!is_started_);
                schedule_thread(l);    // wait and repeat
            }

            if (!result)
                is_terminated_ = true;
        }
        catch (hpx::exception const& e)
        {
            // the lock above might throw yield_aborted
            if (e.get_error() != yield_aborted)
                throw;
        }

        // do not re-schedule this thread
        return threads::thread_result_type(
            threads::thread_schedule_state::terminated,
            threads::invalid_thread_id);
    }

    // schedule a high priority task after a given time interval
    void interval_timer::schedule_thread(std::unique_lock<mutex_type>& l)
    {
        HPX_ASSERT(l.owns_lock());
        HPX_UNUSED(l);

        using namespace hpx::threads;

        error_code ec;

        // create a new suspended thread
        threads::thread_id_type id;
        {
            // FIXME: registering threads might lead to thread suspension since
            // the allocators use hpx::lcos::local::spinlock. Unlocking the
            // lock here would be the right thing but leads to crashes and hangs
            // at shutdown.
            //util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function(util::bind_front(
                    &interval_timer::evaluate, this->shared_from_this())),
                description_.c_str(), threads::thread_priority::boost,
                threads::thread_schedule_hint(),
                threads::thread_stacksize::default_,
                threads::thread_schedule_state::suspended, true);
            id = hpx::threads::register_thread(data, ec);
        }

        if (ec)
        {
            is_terminated_ = true;
            is_started_ = false;
            return;
        }

        // schedule this thread to be run after the given amount of seconds
        threads::thread_id_type timerid =
            threads::set_thread_state(id, std::chrono::microseconds(microsecs_),
                threads::thread_schedule_state::pending,
                threads::thread_restart_state::signaled,
                threads::thread_priority::boost, true, ec);

        if (ec)
        {
            is_terminated_ = true;
            is_started_ = false;

            // abort the newly created thread
            threads::set_thread_state(id,
                threads::thread_schedule_state::pending,
                threads::thread_restart_state::abort,
                threads::thread_priority::boost, true, ec);

            return;
        }

        id_ = id;
        timerid_ = timerid;
        is_started_ = true;
    }
}}}    // namespace hpx::util::detail

namespace hpx { namespace util {
    interval_timer::interval_timer() {}    // -V730

    interval_timer::interval_timer(    // -V730
        util::function_nonser<bool()> const& f, std::int64_t microsecs,
        std::string const& description, bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, microsecs, description, pre_shutdown))
    {
    }

    interval_timer::interval_timer(    // -V730
        util::function_nonser<bool()> const& f,
        util::function_nonser<void()> const& on_term, std::int64_t microsecs,
        std::string const& description, bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, on_term, microsecs, description, pre_shutdown))
    {
    }

    interval_timer::interval_timer(    // -V730
        util::function_nonser<bool()> const& f,
        hpx::chrono::steady_duration const& rel_time, char const* description,
        bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, rel_time.value().count() / 1000, description, pre_shutdown))
    {
    }

    interval_timer::interval_timer(    // -V730
        util::function_nonser<bool()> const& f,
        util::function_nonser<void()> const& on_term,
        hpx::chrono::steady_duration const& rel_time, char const* description,
        bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(f, on_term,
            rel_time.value().count() / 1000, description, pre_shutdown))
    {
    }

    interval_timer::~interval_timer()
    {
        timer_->terminate();
    }

    std::int64_t interval_timer::get_interval() const
    {
        return timer_->get_interval();
    }

    void interval_timer::change_interval(std::int64_t new_interval)
    {
        return timer_->change_interval(new_interval);
    }

    void interval_timer::change_interval(
        hpx::chrono::steady_duration const& new_interval)
    {
        return timer_->change_interval(new_interval.value().count() / 1000);
    }
}}    // namespace hpx::util
