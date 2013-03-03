//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM)
#define HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <string>
#include <vector>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT interval_timer
    {
    public:
        interval_timer();
        interval_timer(HPX_STD_FUNCTION<bool()> const& f,
            boost::int64_t microsecs, std::string const& description, 
            bool pre_shutdown = false);
        interval_timer(HPX_STD_FUNCTION<bool()> const& f,
            HPX_STD_FUNCTION<void()> const& on_term, boost::int64_t microsecs,
                std::string const& description, bool pre_shutdown = false);
        ~interval_timer();

        bool start();
        bool stop();

        bool is_started() const { return is_started_; }
        bool is_terminated() const { return is_terminated_; }

        boost::int64_t get_interval() const { return microsecs_; }

    protected:
        // schedule a high priority task after a given time interval
        void schedule_thread();

        threads::thread_state_enum
            evaluate(threads::thread_state_ex_enum statex);

        void terminate();             // handle system shutdown
        bool stop_locked();

    private:
        typedef lcos::local::spinlock mutex_type;

        mutable mutex_type mtx_;
        HPX_STD_FUNCTION<bool()> f_;  ///< function to call
        HPX_STD_FUNCTION<void()> on_term_;  ///< function to call on termination
        boost::int64_t microsecs_;       ///< time interval
        threads::thread_id_type id_;  ///< id of currently scheduled thread
        std::string description_;     ///< description of this interval timer

        bool pre_shutdown_;           ///< execute termination during pre-shutdown
        bool is_started_;             ///< timer has been started (is running)
        bool first_start_;            ///< flag to distinguish first invocation of start()
        bool is_terminated_;          ///< The timer has been terminated
    };
}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif
