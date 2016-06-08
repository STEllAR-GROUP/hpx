//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM)
#define HPX_UTIL_INTERVAL_TIMER_SEP_27_2011_0434PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{
    class interval_timer;
}}

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT interval_timer
      : public std::enable_shared_from_this<interval_timer>
    {
    private:
        friend class util::interval_timer;

        typedef lcos::local::spinlock mutex_type;

    public:
        interval_timer();
        interval_timer(util::function_nonser<bool()> const& f,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::int64_t microsecs, std::string const& description,
            bool pre_shutdown);

        ~interval_timer();

        bool start(bool evaluate);
        bool stop();

        bool restart(bool evaluate);

        bool is_started() const { return is_started_; }
        bool is_terminated() const { return is_terminated_; }

        boost::int64_t get_interval() const;

        void slow_down(boost::int64_t max_interval);
        void speed_up(boost::int64_t min_interval);

    protected:
        // schedule a high priority task after a given time interval
        void schedule_thread(std::unique_lock<mutex_type> & l);

        threads::thread_state_enum
            evaluate(threads::thread_state_ex_enum statex);

        void terminate();             // handle system shutdown
        bool stop_locked();

    private:
        mutable mutex_type mtx_;
        util::function_nonser<bool()> f_; ///< function to call
        util::function_nonser<void()> on_term_; ///< function to call on termination
        boost::int64_t microsecs_;    ///< time interval
        threads::thread_id_type id_;  ///< id of currently scheduled thread
        std::string description_;     ///< description of this interval timer

        bool pre_shutdown_;           ///< execute termination during pre-shutdown
        bool is_started_;             ///< timer has been started (is running)
        bool first_start_;
        ///^ flag to distinguish first invocation of start()
        bool is_terminated_;          ///< The timer has been terminated
        bool is_stopped_;
    };
}}}

namespace hpx { namespace util
{
    class HPX_EXPORT interval_timer
    {
        HPX_MOVABLE_ONLY(interval_timer);

    public:
        interval_timer();
        interval_timer(util::function_nonser<bool()> const& f,
            boost::int64_t microsecs, std::string const& description = "",
            bool pre_shutdown = false);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::int64_t microsecs, std::string const& description = "",
            bool pre_shutdown = false);

        interval_timer(util::function_nonser<bool()> const& f,
            util::steady_duration const& rel_time,
            char const*  description = "", bool pre_shutdown = false);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            util::steady_duration const& rel_time,
            char const*  description = "", bool pre_shutdown = false);

        ~interval_timer();

        bool start(bool evaluate = true)
        {
            return timer_->start(evaluate);
        }
        bool stop()
        {
            return timer_->stop();
        }

        bool restart(bool evaluate = true)
        {
            return timer_->restart(evaluate);
        }

        bool is_started() const
        {
            return timer_->is_started();
        }
        bool is_terminated() const
        {
            return timer_->is_terminated();
        }

        boost::int64_t get_interval() const;

        void slow_down(boost::int64_t max_interval);
        void speed_up(boost::int64_t min_interval);

        void slow_down(util::steady_duration const& max_interval);
        void speed_up(util::steady_duration const& min_interval);

    private:
        std::shared_ptr<detail::interval_timer> timer_;
    };
}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif
