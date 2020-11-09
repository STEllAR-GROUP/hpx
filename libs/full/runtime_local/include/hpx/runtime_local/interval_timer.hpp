//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hpx { namespace util {
    class interval_timer;
}}    // namespace hpx::util

namespace hpx { namespace util { namespace detail {
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
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::int64_t microsecs, std::string const& description,
            bool pre_shutdown);

        ~interval_timer();

        bool start(bool evaluate);
        bool stop(bool terminate = false);

        bool restart(bool evaluate);

        bool is_started() const
        {
            return is_started_;
        }
        bool is_terminated() const
        {
            return is_terminated_;
        }

        std::int64_t get_interval() const;

        void change_interval(std::int64_t new_interval);

    protected:
        // schedule a high priority task after a given time interval
        void schedule_thread(std::unique_lock<mutex_type>& l);

        threads::thread_result_type evaluate(
            threads::thread_restart_state statex);

        void terminate();    // handle system shutdown
        bool stop_locked();

    private:
        mutable mutex_type mtx_;
        util::function_nonser<bool()> f_;    ///< function to call
        util::function_nonser<void()>
            on_term_;                   ///< function to call on termination
        std::int64_t microsecs_;        ///< time interval
        threads::thread_id_type id_;    ///< id of currently scheduled thread
        threads::thread_id_type timerid_;    ///< id of the timer thread for the
                                             ///< currently scheduled thread
        std::string description_;    ///< description of this interval timer

        bool pre_shutdown_;    ///< execute termination during pre-shutdown
        bool is_started_;      ///< timer has been started (is running)
        bool first_start_;
        ///^ flag to distinguish first invocation of start()
        bool is_terminated_;    ///< The timer has been terminated
        bool is_stopped_;
    };
}}}    // namespace hpx::util::detail

namespace hpx { namespace util {
    class HPX_EXPORT interval_timer
    {
    public:
        HPX_NON_COPYABLE(interval_timer);

    public:
        interval_timer();
        interval_timer(util::function_nonser<bool()> const& f,
            std::int64_t microsecs, std::string const& description = "",
            bool pre_shutdown = false);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            std::int64_t microsecs, std::string const& description = "",
            bool pre_shutdown = false);

        interval_timer(util::function_nonser<bool()> const& f,
            hpx::chrono::steady_duration const& rel_time,
            char const* description = "", bool pre_shutdown = false);
        interval_timer(util::function_nonser<bool()> const& f,
            util::function_nonser<void()> const& on_term,
            hpx::chrono::steady_duration const& rel_time,
            char const* description = "", bool pre_shutdown = false);

        ~interval_timer();

        bool start(bool evaluate = true)
        {
            return timer_->start(evaluate);
        }
        bool stop(bool terminate = false)
        {
            return timer_->stop(terminate);
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

        std::int64_t get_interval() const;

        void change_interval(std::int64_t new_interval);

        void change_interval(hpx::chrono::steady_duration const& new_interval);

    private:
        std::shared_ptr<detail::interval_timer> timer_;
    };
}}    // namespace hpx::util

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
