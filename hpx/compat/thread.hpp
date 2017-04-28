//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/chrono/chrono.hpp
// hpxinspect:nodeprecatedname:boost::chrono
// hpxinspect:nodeprecatedname:boost::thread
// hpxinspect:nodeprecatedname:boost::this_thread::get_id
// hpxinspect:nodeprecatedname:boost::this_thread::yield
// hpxinspect:nodeprecatedname:boost::this_thread::sleep_until
// hpxinspect:nodeprecatedname:boost::this_thread::sleep_for

#ifndef HPX_COMPAT_THREAD_HPP
#define HPX_COMPAT_THREAD_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
#include <boost/chrono/chrono.hpp>
#include <boost/thread/thread.hpp>

#include <chrono>
#include <cstdint>
#include <utility>

namespace hpx { namespace compat
{
    class thread : private boost::thread
    {
        using base_type = boost::thread;

    public:
        // types:
        using base_type::id;
        using base_type::native_handle_type;

        // construct/copy/destroy:
        thread() HPX_NOEXCEPT
          : base_type()
        {}

        template <typename F, typename ...Args>
        explicit thread(F&& f, Args&&... args)
          : base_type(std::forward<F>(f), std::forward<Args>(args)...)
        {}

        // members:
        using base_type::swap;
        using base_type::joinable;
        using base_type::join;
        using base_type::detach;
        using base_type::get_id;
        using base_type::native_handle;

        // static members:
        using base_type::hardware_concurrency;
    };

    namespace this_thread
    {
        using boost::this_thread::get_id;

        using boost::this_thread::yield;

        template <typename Clock, typename Duration>
        void sleep_until(std::chrono::time_point<Clock, Duration> const& abs_time)
        {
            using boost_ratio =
                boost::ratio<Duration::period::num, Duration::period::den>;
            using boost_duration =
                boost::chrono::duration<typename Duration::rep, boost_ratio>;
            using boost_time_point =
                boost::chrono::time_point<Clock, boost_duration>;

            boost::this_thread::sleep_until(boost_time_point(
                boost_duration(abs_time.time_since_epoch().count())));
        }

        template <typename Rep, typename Period>
        void sleep_for(std::chrono::duration<Rep, Period> const& rel_time)
        {
            using boost_ratio = boost::ratio<Period::num, Period::den>;
            using boost_duration = boost::chrono::duration<Rep, boost_ratio>;

            boost::this_thread::sleep_for(boost_duration(rel_time.count()));
        }
    }
}}
#else
///////////////////////////////////////////////////////////////////////////////
#include <thread>

namespace hpx { namespace compat
{
    using thread = std::thread;

    namespace this_thread = std::this_thread;
}}
#endif

#endif /*HPX_COMPAT_THREAD_HPP*/
