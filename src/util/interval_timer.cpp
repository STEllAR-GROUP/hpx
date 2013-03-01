//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/bind.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    interval_timer::interval_timer()
      : microsecs_(0), id_(0)
    {}

    interval_timer::interval_timer(HPX_STD_FUNCTION<bool()> const& f,
            HPX_STD_FUNCTION<void()> const& on_term,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(on_term),
        microsecs_(microsecs), id_(0), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false)
    {}

    interval_timer::interval_timer(HPX_STD_FUNCTION<bool()> const& f,
            HPX_STD_FUNCTION<void()> const& on_term,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown)
      : f_(f), on_term_(on_term),
        microsecs_(microsecs), id_(0), description_(description),
        pre_shutdown_(pre_shutdown), is_started_(false), first_start_(true),
        is_terminated_(false)
    {}

    bool interval_timer::start()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!is_started_) {
            is_started_ = true;

            if (first_start_) {
                first_start_ = false;
                l.unlock();
                if (pre_shutdown_)
                    register_pre_shutdown_function(boost::bind(&interval_timer::terminate, this));
                else
                    register_shutdown_function(boost::bind(&interval_timer::terminate, this));
            }
            else {
                l.unlock();
            }

            evaluate(threads::wait_signaled);
            return true;
        }
        return false;
    }

    bool interval_timer::stop()
    {
        mutex_type::scoped_lock l(mtx_);
        return stop_locked();
    }

    bool interval_timer::stop_locked()
    {
        if (is_started_) {
            is_started_ = false;

            if (id_) {
                error_code ec(lightweight);       // avoid throwing on error
                threads::set_thread_state(id_, threads::pending,
                    threads::wait_abort, threads::thread_priority_critical, ec);
                id_ = 0;
            }
            return true;
        }
        BOOST_ASSERT(id_ == 0);
        return false;
    }

    void interval_timer::terminate()
    {
        mutex_type::scoped_lock l(mtx_);
        if (!is_terminated_) {
            is_terminated_ = true;
            if (on_term_)
                on_term_();
            stop_locked();
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
        if (statex == threads::wait_abort || 0 == microsecs_)
            return threads::terminated;        // object has been finalized, exit

        mutex_type::scoped_lock l(mtx_);
        id_ = 0;

        bool result = false;

        {
            util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            result = f_();            // invoke the supplied function
        }

        if (result)
            schedule_thread();        // wait and repeat
        return threads::terminated;   // do not re-schedule this thread
    }

    // schedule a high priority task after a given time interval
    void interval_timer::schedule_thread()
    {
        using namespace hpx::threads;

        // create a new suspended thread
        id_ = hpx::applier::register_thread_plain(
            boost::bind(&interval_timer::evaluate, this, _1),
            description_.c_str(), threads::suspended, true,
            threads::thread_priority_critical);

        // schedule this thread to be run after the given amount of seconds
        threads::set_thread_state(id_,
            boost::posix_time::microseconds(microsecs_),
            threads::pending, threads::wait_signaled,
            threads::thread_priority_critical);
    }
}}

