//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/register_locks_globally.hpp>
#include <hpx/util/static.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/ptr_container/ptr_map.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
#ifdef HPX_HAVE_VERIFY_LOCKS_GLOBALLY
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct global_lock_data
        {
#ifdef HPX_HAVE_VERIFY_LOCKS_BACKTRACE
            global_lock_data()
            {
                backtrace_ = hpx::detail::backtrace_direct(75);
            }

            std::string backtrace_;
#endif
        };

        ///////////////////////////////////////////////////////////////////////
        struct register_locks_globally
        {
            typedef lcos::local::spinlock mutex_type;
            typedef boost::ptr_map<void const*, global_lock_data> held_locks_map;

            struct global_locks_data
            {
                typedef lcos::local::spinlock mutex_type;

                mutable mutex_type mtx_;
                held_locks_map held_locks_;
            };

            struct tls_tag {};

            static global_locks_data& get_global_locks_data()
            {
                hpx::util::static_<global_locks_data, tls_tag> held_locks;
                return held_locks.get();
            }

            static bool lock_detection_enabled_;

            static held_locks_map& get_lock_map()
            {
                return get_global_locks_data().held_locks_;
            }

            static mutex_type& get_mutex()
            {
                return get_global_locks_data().mtx_;
            }
        };

        bool register_locks_globally::lock_detection_enabled_ = false;
    }

    ///////////////////////////////////////////////////////////////////////////
    void enable_global_lock_detection()
    {
        detail::register_locks_globally::lock_detection_enabled_ = true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool register_lock_globally(void const* lock)
    {
        using detail::register_locks_globally;

        if (register_locks_globally::lock_detection_enabled_
            && 0 != threads::get_self_ptr())
        {
            register_locks_globally::held_locks_map& held_locks =
                register_locks_globally::get_lock_map();

            register_locks_globally::mutex_type::scoped_lock l(
                register_locks_globally::get_mutex());

            register_locks_globally::held_locks_map::iterator it =
                held_locks.find(lock);
            if (it != held_locks.end())
                return false;     // this lock is already registered

            std::pair<register_locks_globally::held_locks_map::iterator, bool> p;
            p = held_locks.insert(lock, new detail::global_lock_data());

            return p.second;
        }
        return true;
    }

    // unregister the given lock from this HPX-thread
    bool unregister_lock_globally(void const* lock)
    {
        using detail::register_locks_globally;

        if (register_locks_globally::lock_detection_enabled_
            && 0 != threads::get_self_ptr())
        {
            register_locks_globally::held_locks_map& held_locks =
                register_locks_globally::get_lock_map();

            register_locks_globally::mutex_type::scoped_lock l(
                register_locks_globally::get_mutex());

            register_locks_globally::held_locks_map::iterator it =
                held_locks.find(lock);
            if (it == held_locks.end())
                return false;           // this lock is not registered

            held_locks.erase(lock);
        }
        return true;
    }

#else

    void enable_global_lock_detection()
    {
    }

    bool register_lock_globally(void const*)
    {
        return true;
    }

    bool unregister_lock_globally(void const*)
    {
        return true;
    }

#endif
}}


