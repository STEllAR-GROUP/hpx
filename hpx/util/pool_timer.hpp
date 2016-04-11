//  Copyright (c) 2016 Bibek Wagle
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_POOL_TIMER)
#define HPX_UTIL_POOL_TIMER

#include <hpx/config.hpp>
#include <hpx/util/function.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <boost/asio/deadline_timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


namespace hpx { namespace util
{
    class pool_timer;
}}

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT pool_timer
      : public boost::enable_shared_from_this<pool_timer>
    {
    private:
        friend class util::pool_timer;

        typedef lcos::local::spinlock mutex_type;

    public:
        pool_timer();
        pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::posix_time::time_duration microsecs,
            std::string const& description,
            bool pre_shutdown);

        ~pool_timer();

        bool start(bool evaluate);
        bool stop();

        bool is_started() const { return is_started_; }
        bool is_terminated() const { return is_terminated_; }
        void timer_handler();

        void terminate();             // handle system shutdown
        bool stop_locked();

    private:
        mutable mutex_type mtx_;
        util::function_nonser<bool()> f_; ///< function to call
        util::function_nonser<void()> on_term_; ///< function to call on termination
        boost::posix_time::time_duration microsecs_;    ///< time interval
        std::string description_;     ///< description of this interval timer

        bool pre_shutdown_;           ///< execute termination during pre-shutdown
        bool is_started_;             ///< timer has been started (is running)
        bool first_start_;
        ///^ flag to distinguish first invocation of start()
        bool is_terminated_;          ///< The timer has been terminated
        bool is_stopped_;
        boost::asio::deadline_timer mytimer_;
    };
}}}

namespace hpx { namespace util
{
    class HPX_EXPORT pool_timer
    {
        HPX_MOVABLE_ONLY(pool_timer);

    public:
        pool_timer();

        pool_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            boost::posix_time::time_duration microsecs,
            std::string const& description = "",
            bool pre_shutdown = false);

        ~pool_timer();

        bool start(bool evaluate = true)
        {
            return timer_->start(evaluate);
        }
        bool stop()
        {
            return timer_->stop();
        }

        bool is_started() const
        {
            return timer_->is_started();
        }
        bool is_terminated() const
        {
            return timer_->is_terminated();
        }

    private:
        boost::shared_ptr<detail::pool_timer> timer_;
    };
}}


#endif
