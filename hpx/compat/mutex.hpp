//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:boost::mutex
// hpxinspect:nodeprecatedname:boost::recursive_mutex
// hpxinspect:nodeprecatedname:boost::once_flag
// hpxinspect:nodeprecatedname:boost::call_once

#ifndef HPX_COMPAT_MUTEX_HPP
#define HPX_COMPAT_MUTEX_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
#include <boost/thread/mutex.hpp>
#include <boost/thread/once.hpp>
#include <boost/thread/recursive_mutex.hpp>

#include <cstdint>
#include <utility>

namespace hpx { namespace compat
{
    class mutex : private boost::mutex
    {
        friend class condition_variable;

        using base_type = boost::mutex;

    public:
        mutex() noexcept
          : base_type()
        {}

        using base_type::lock;
        using base_type::try_lock;
        using base_type::unlock;
    };

    class recursive_mutex : private boost::recursive_mutex
    {
        using base_type = boost::recursive_mutex;

    public:
        recursive_mutex() noexcept
          : base_type()
        {}

        using base_type::lock;
        using base_type::try_lock;
        using base_type::unlock;
    };

    struct once_flag : boost::once_flag
    {
        HPX_CONSTEXPR once_flag() noexcept
          : boost::once_flag(BOOST_ONCE_INIT)
        {}
    };

    using boost::call_once;
}}
#else
///////////////////////////////////////////////////////////////////////////////
#include <mutex>

namespace hpx { namespace compat
{
    using mutex = std::mutex;
    using recursive_mutex = std::recursive_mutex;

    using once_flag = std::once_flag;
    using std::call_once;
}}
#endif

#endif /*HPX_COMPAT_MUTEX_HPP*/
