//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_REGISTER_LOCKS_JUN_26_2012_1029AM)
#define HPX_UTIL_REGISTER_LOCKS_JUN_26_2012_1029AM

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX11_SFINAE_EXPR)
#include <hpx/util/always_void.hpp>
#else
#include <boost/thread/locks.hpp>
#endif

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    struct register_lock_data {};

    // Always provide function exports, which guarantees ABI compatibility of
    // Debug and Release builds.

    template <typename Lock, typename Enable = void>
    struct ignore_while_checking;

#if defined(HPX_HAVE_VERIFY_LOCKS) || defined(HPX_EXPORTS)

    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT bool register_lock(void const* lock,
        register_lock_data* data = 0);
    HPX_API_EXPORT bool unregister_lock(void const* lock);
    HPX_API_EXPORT void verify_no_locks();
    HPX_API_EXPORT void force_error_on_lock();
    HPX_API_EXPORT void enable_lock_detection();
    HPX_API_EXPORT void ignore_lock(void const* lock);
    HPX_API_EXPORT void reset_ignored(void const* lock);
    HPX_API_EXPORT void ignore_all_locks();
    HPX_API_EXPORT void reset_ignored_all();

    ///////////////////////////////////////////////////////////////////////////
    struct ignore_all_while_checking
    {
        ignore_all_while_checking()
        {
            ignore_all_locks();
        }

        ~ignore_all_while_checking()
        {
            reset_ignored_all();
        }
    };

#if defined(HPX_HAVE_CXX11_SFINAE_EXPR)
    template <typename Lock>
    struct ignore_while_checking<Lock,
        typename util::always_void<
            decltype(std::declval<Lock>().mutex())
        >::type>
    {
        ignore_while_checking(Lock const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };
#else
    template <typename Mutex>
    struct ignore_while_checking<boost::unique_lock<Mutex> >
    {
        ignore_while_checking(boost::unique_lock<Mutex> const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };

    template <typename Mutex>
    struct ignore_while_checking<boost::upgrade_lock<Mutex> >
    {
        ignore_while_checking(boost::upgrade_lock<Mutex> const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };

    template <typename Mutex>
    struct ignore_while_checking<boost::shared_lock<Mutex> >
    {
        ignore_while_checking(boost::shared_lock<Mutex> const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };

    template <typename Mutex>
    struct ignore_while_checking<boost::detail::try_lock_wrapper<Mutex> >
    {
        ignore_while_checking(
                boost::detail::try_lock_wrapper<Mutex> const* lock)
          : mtx_(lock->mutex())
        {
            ignore_lock(mtx_);
        }

        ~ignore_while_checking()
        {
            reset_ignored(mtx_);
        }

        void const* mtx_;
    };
#endif

#else
    template <typename Lock, typename Enable>
    struct ignore_while_checking
    {
        ignore_while_checking(void const* lock) {}
    };

    struct ignore_all_while_checking
    {
        ignore_all_while_checking() {}
    };

    inline bool register_lock(void const*, util::register_lock_data* = 0)
    {
        return true;
    }
    inline bool unregister_lock(void const*)
    {
        return true;
    }
    inline void verify_no_locks()
    {
    }
    inline void force_error_on_lock()
    {
    }
    inline void enable_lock_detection()
    {
    }
    inline void ignore_lock(void const* /*lock*/)
    {
    }
    inline void reset_ignored(void const* /*lock*/)
    {
    }

    inline void ignore_all_locks()
    {
    }
    inline void reset_ignored_all()
    {
    }
#endif
}}

#endif

