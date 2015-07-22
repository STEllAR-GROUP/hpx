//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/bind.hpp>
#include <boost/thread/locks.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    interval_timer::interval_timer()
      : microsecs_(0), id_(0)
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(),
        microsecs_(microsecs), id_(0), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false), is_stopped_(false)
    {}

    interval_timer::interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(on_term),
        microsecs_(microsecs), id_(0), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false), is_stopped_(false)
    {}

    bool interval_timer::start(bool evaluate_)
    {
        boost::unique_lock<mutex_type> l(mtx_);
        if (is_terminated_)
            return false;

        if (!is_started_) {
            if (first_start_) {
                first_start_ = false;

                util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                if (pre_shutdown_)
                    register_pre_shutdown_function(boost::bind(&interval_timer::terminate, this));
                else
                    register_shutdown_function(boost::bind(&interval_timer::terminate, this));
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

        boost::unique_lock<mutex_type> l(mtx_);

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
        boost::lock_guard<mutex_type> l(mtx_);
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
                id_ = 0;
            }
            return true;
        }

        HPX_ASSERT(id_ == 0);
        return false;
    }

    void interval_timer::terminate()
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

    interval_timer::~interval_timer()
    {
        try {
            terminate();
        }
        catch(...) {
            ;   // there is nothing we can do here
        }
    }

    threads::thread_state_enum interval_timer::evaluate(
        threads::thread_state_ex_enum statex)
    {
        try {
            boost::unique_lock<mutex_type> l(mtx_);

            if (is_stopped_ || is_terminated_ ||
                statex == threads::wait_abort || 0 == microsecs_)
            {
                return threads::terminated;        // object has been finalized, exit
            }

            if (id_ != 0 && id_ != threads::get_self_id())
                return threads::terminated;        // obsolete timer thread

            id_ = 0;
            is_started_ = false;

            bool result = false;

            {
                util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                result = f_();            // invoke the supplied function
            }

            // some other thread might already have started the timer
            if (0 == id_ && result) {
                HPX_ASSERT(!is_started_);
                schedule_thread(l);        // wait and repeat
            }
        }
        catch (hpx::exception const& e){
            // the lock above might throw yield_aborted
            if (e.get_error() != yield_aborted)
                throw;
        }
        return threads::terminated;   // do not re-schedule this thread
    }

    // schedule a high priority task after a given time interval
    void interval_timer::schedule_thread(boost::unique_lock<mutex_type> & l)
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
            //util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
            id = hpx::applier::register_thread_plain(
                boost::bind(&interval_timer::evaluate, this, _1),
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
            boost::chrono::microseconds(microsecs_),
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
}}

