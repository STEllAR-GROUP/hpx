//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNLOCK_GUARD_HPP
#define HPX_UTIL_UNLOCK_GUARD_HPP

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // This is a helper structure to make sure a lock gets unlocked and locked
    // again in a scope.
    template <typename Mutex>
    class unlock_guard
    {
    public:
        HPX_NON_COPYABLE(unlock_guard);

    public:
        typedef Mutex mutex_type;

        explicit unlock_guard(Mutex& m)
          : m_(m)
        {
            m_.unlock();
        }

        ~unlock_guard()
        {
            m_.lock();
        }

    private:
        Mutex& m_;
    };

    template <typename Mutex>
    class unlock_guard_try
    {
    public:
        HPX_NON_COPYABLE(unlock_guard_try);

    public:
        typedef Mutex mutex_type;

        explicit unlock_guard_try(Mutex& m)
          : m_(m)
        {
            m_.unlock();
        }

        ~unlock_guard_try()
        {
            m_.try_lock();
        }

    private:
        Mutex& m_;
    };
}}

#endif /*HPX_UTIL_UNLOCK_GUARD_HPP*/
