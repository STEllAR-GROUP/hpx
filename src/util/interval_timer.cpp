//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind_front.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    interval_timer::interval_timer()
      : microsecs_(0), id_(nullptr)
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(),
        microsecs_(microsecs), id_(nullptr), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false), is_stopped_(false)
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(on_term),
        microsecs_(microsecs), id_(nullptr), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false), is_stopped_(false)
    {}

    bool interval_timer::start(bool evaluate_)
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (is_terminated_)
            return false;

        if (!is_started_) {
            if (first_start_) {
                first_start_ = false;

                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                if (pre_shutdown_)
                {
                    register_pre_shutdown_function(
                        util::deferred_call(&interval_timer::terminate,
                            this->shared_from_this()));
                }
                else
                {
                    register_shutdown_function(
                        util::deferred_call(&interval_timer::terminate,
                            this->shared_from_this()));
                }
            }

            is_stopped_ = false;

            if (evaluate_) {
                l.unlock();
                evaluate(threads::wait_signaled);
            }
            else {
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
        if (evaluate_) {
            l.unlock();
            evaluate(threads::wait_signaled);
        }
        else {
            schedule_thread(l);
        }
        return true;
    }

    bool interval_timer::stop()
    {
        std::lock_guard<mutex_type> l(mtx_);
        is_stopped_ = true;
        return stop_locked();
    }

    bool interval_timer::stop_locked()
    {
        if (is_started_) {
            is_started_ = false;

            if (id_) {
                error_code ec(lightweight);       // avoid throwing on error
                threads::set_thread_state(id_, threads::pending,
                    threads::wait_abort, threads::thread_priority_boost, ec);
                id_.reset();
            }
            return true;
        }

        HPX_ASSERT(id_ == nullptr);
        return false;
    }

    void interval_timer::terminate()
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (!is_terminated_) {
            is_terminated_ = true;
            stop_locked();

            if (on_term_) {
                l.unlock();
                on_term_();
            }
        }
    }

    interval_timer::~interval_timer()
    {
        try {
            terminate();
        }
        catch(...) {
            ;   // there is nothing we can do here
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
        threads::thread_state_ex_enum statex)
    {
        try {
            std::unique_lock<mutex_type> l(mtx_);

            if (is_stopped_ || is_terminated_ ||
                statex == threads::wait_abort || 0 == microsecs_)
            {
                // object has been finalized, exit
                return threads::thread_result_type(threads::terminated,
                    threads::invalid_thread_id);
            }

            if (id_ != nullptr && id_ != threads::get_self_id())
            {
                // obsolete timer thread
                return threads::thread_result_type(threads::terminated,
                    threads::invalid_thread_id);
            }

            id_.reset();
            is_started_ = false;

            bool result = false;

            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                result = f_();            // invoke the supplied function
            }

            // some other thread might already have started the timer
            if (nullptr == id_ && result) {
                HPX_ASSERT(!is_started_);
                schedule_thread(l);        // wait and repeat
            }

            if (!result)
                is_terminated_ = true;
        }
        catch (hpx::exception const& e){
            // the lock above might throw yield_aborted
            if (e.get_error() != yield_aborted)
                throw;
        }

        // do not re-schedule this thread
        return threads::thread_result_type(threads::terminated,
            threads::invalid_thread_id);
    }

    // schedule a high priority task after a given time interval
    void interval_timer::schedule_thread(std::unique_lock<mutex_type> & l)
    {
        HPX_ASSERT(l.owns_lock());

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
            id = hpx::applier::register_thread_plain(
                util::bind_front(&interval_timer::evaluate,
                    this->shared_from_this()),
                description_.c_str(), threads::suspended, true,
                threads::thread_priority_boost, std::size_t(-1),
                threads::thread_stacksize_default, ec);
        }

        if (ec) {
            is_terminated_ = true;
            is_started_ = false;
            return;
        }

        // schedule this thread to be run after the given amount of seconds
        threads::set_thread_state(id,
            std::chrono::microseconds(microsecs_),
            threads::pending, threads::wait_signaled,
            threads::thread_priority_boost, ec);

        if (ec) {
            is_terminated_ = true;
            is_started_ = false;

            // abort the newly created thread
            threads::set_thread_state(id, threads::pending, threads::wait_abort,
                threads::thread_priority_boost, ec);

            return;
        }

        id_ = id;
        is_started_ = true;
    }
}}}

namespace hpx { namespace util
{
    interval_timer::interval_timer() {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, microsecs, description, pre_shutdown))
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, on_term, microsecs, description, pre_shutdown))
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            util::steady_duration const& rel_time,
            char const*  description, bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, rel_time.value().count() / 1000, description, pre_shutdown))
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            util::steady_duration const& rel_time,
            char const*  description, bool pre_shutdown)
      : timer_(std::make_shared<detail::interval_timer>(
            f, on_term, rel_time.value().count() / 1000, description,
            pre_shutdown))
    {}

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

    void interval_timer::change_interval(util::steady_duration const& new_interval)
    {
        return timer_->change_interval(new_interval.value().count() / 1000);
    }
}}
