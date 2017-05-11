//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/chrono/chrono.hpp
// hpxinspect:nodeprecatedinclude:boost/thread/locks.hpp
// hpxinspect:nodeprecatedname:boost::chrono
// hpxinspect:nodeprecatedname:boost::unique_lock
// hpxinspect:nodeprecatedname:boost::cv_status
// hpxinspect:nodeprecatedname:boost::condition_variable
// hpxinspect:nodeprecatedname:boost::mutex

#ifndef HPX_COMPAT_CONDITION_VARIABLE_HPP
#define HPX_COMPAT_CONDITION_VARIABLE_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
#include <hpx/compat/mutex.hpp>

#include <boost/thread/locks.hpp>
#include <boost/thread/condition_variable.hpp>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>

namespace hpx { namespace compat
{
    using cv_status = boost::cv_status;

    class condition_variable : private boost::condition_variable
    {
        using base_type = boost::condition_variable;

        struct transformed_lock
        {
            explicit transformed_lock(std::unique_lock<mutex>& lock)
              : lock(lock)
              , boost_lock(
                    *static_cast<boost::mutex*>(lock.release()),
                    boost::adopt_lock)
            {}

            ~transformed_lock()
            {
                lock = std::unique_lock<mutex>(
                    *static_cast<mutex*>(boost_lock.release()),
                    std::adopt_lock);
            }

            boost::unique_lock<boost::mutex>& get() HPX_NOEXCEPT
            {
                return boost_lock;
            }

        private:
            std::unique_lock<mutex>& lock;
            boost::unique_lock<boost::mutex> boost_lock;
        };

    public:
        condition_variable()
          : base_type()
        {}

        using base_type::notify_one;
        using base_type::notify_all;

        void wait(std::unique_lock<mutex>& lock)
        {
            transformed_lock boost_lock(lock);

            return base_type::wait(boost_lock.get());
        }

        template <typename Predicate>
        void wait(std::unique_lock<mutex>& lock, Predicate&& pred)
        {
            transformed_lock boost_lock(lock);

            return base_type::wait(boost_lock.get(), std::forward<Predicate>(pred));
        }

        template <typename Clock, typename Duration>
        cv_status wait_until(
            std::unique_lock<mutex>& lock,
            std::chrono::time_point<Clock, Duration> const& abs_time)
        {
            transformed_lock boost_lock(lock);

            using boost_ratio =
                boost::ratio<Duration::period::num, Duration::period::den>;
            using boost_duration =
                boost::chrono::duration<typename Duration::rep, boost_ratio>;
            using boost_time_point =
                boost::chrono::time_point<Clock, boost_duration>;

            return base_type::wait_until(boost_lock.get(),
                boost_time_point(boost_duration(abs_time.time_since_epoch().count())));
        }

        template <typename Clock, typename Duration, typename Predicate>
        bool wait_until(
            std::unique_lock<mutex>& lock,
            std::chrono::time_point<Clock, Duration> const& abs_time,
            Predicate&& pred)
        {
            transformed_lock boost_lock(lock);

            using boost_ratio =
                boost::ratio<Duration::period::num, Duration::period::den>;
            using boost_duration =
                boost::chrono::duration<typename Duration::rep, boost_ratio>;
            using boost_time_point =
                boost::chrono::time_point<Clock, boost_duration>;

            return base_type::wait_until(boost_lock.get(),
                boost_time_point(boost_duration(abs_time.time_since_epoch().count())),
                std::forward<Predicate>(pred));
        }

        template <typename Rep, typename Period>
        cv_status wait_for(
            std::unique_lock<mutex>& lock,
            std::chrono::duration<Rep, Period> const& rel_time)
        {
            transformed_lock boost_lock(lock);

            using boost_ratio = boost::ratio<Period::num, Period::den>;
            using boost_duration = boost::chrono::duration<Rep, boost_ratio>;

            return base_type::wait_for(boost_lock.get(),
                boost_duration(rel_time.count()));
        }

        template <typename Rep, typename Period, typename Predicate>
        bool wait_for(
            std::unique_lock<mutex>& lock,
            std::chrono::duration<Rep, Period> const& rel_time,
            Predicate&& pred)
        {
            transformed_lock boost_lock(lock);

            using boost_ratio = boost::ratio<Period::num, Period::den>;
            using boost_duration = boost::chrono::duration<Rep, boost_ratio>;

            return base_type::wait_for(boost_lock.get(),
                boost_duration(rel_time.count()),
                std::forward<Predicate>(pred));
        }
    };
}}
#else
///////////////////////////////////////////////////////////////////////////////
#include <condition_variable>

namespace hpx { namespace compat
{
    using cv_status = std::cv_status;
    using condition_variable = std::condition_variable;
}}
#endif

#endif /*HPX_COMPAT_CONDITION_VARIABLE_HPP*/
