//  Copyright (c) 2012 Hartmut Kaiser
//
//  taken from:
//  boost/detail/spinlock_pool.hpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_SPINLOCK_POOL_HPP_INCLUDED
#define HPX_UTIL_SPINLOCK_POOL_HPP_INCLUDED

// MS compatible compilers support #pragma once

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/config.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

#include <boost/smart_ptr/detail/spinlock.hpp>
#include <boost/version.hpp>

#include <cstddef>

namespace hpx { namespace util
{
#if HPX_HAVE_ITTNOTIFY != 0
    namespace detail
    {
        template <typename Tag>
        struct itt_spinlock_init
        {
            itt_spinlock_init();
            ~itt_spinlock_init();
        };
    }
#endif

    template <typename Tag>
    class spinlock_pool
    {
    private:
        static boost::detail::spinlock pool_[ 41 ];
#if HPX_HAVE_ITTNOTIFY != 0
        static detail::itt_spinlock_init<Tag> init_;
#endif

    public:

        static boost::detail::spinlock & spinlock_for( void const * pv )
        {
            std::size_t i = reinterpret_cast< std::size_t >( pv ) % 41;
            return pool_[ i ];
        }

        class scoped_lock
        {
        private:
            boost::detail::spinlock & sp_;

            HPX_NON_COPYABLE(scoped_lock);

        public:
            explicit scoped_lock( void const * pv ): sp_( spinlock_for( pv ) )
            {
                lock();
            }

            ~scoped_lock()
            {
                unlock();
            }

            void lock()
            {
                HPX_ITT_SYNC_PREPARE(&sp_);
                sp_.lock();
                HPX_ITT_SYNC_ACQUIRED(&sp_);
                util::register_lock(&sp_);
            }

            void unlock()
            {
                HPX_ITT_SYNC_RELEASING(&sp_);
                sp_.unlock();
                HPX_ITT_SYNC_RELEASED(&sp_);
                util::unregister_lock(&sp_);
            }
        };
    };

    template <typename Tag>
    boost::detail::spinlock spinlock_pool<Tag>::pool_[ 41 ] =
    {
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT, BOOST_DETAIL_SPINLOCK_INIT,
        BOOST_DETAIL_SPINLOCK_INIT
    };

#if HPX_HAVE_ITTNOTIFY != 0
    namespace detail
    {
        template <typename Tag>
        itt_spinlock_init<Tag>::itt_spinlock_init()
        {
            for (int i = 0; i < 41; ++i)
            {
                HPX_ITT_SYNC_CREATE(&spinlock_pool<Tag>::pool_[i],
                    "boost::detail::spinlock", 0);
                HPX_ITT_SYNC_RENAME(&spinlock_pool<Tag>::pool_[i],
                    "boost::detail::spinlock");
            }
        }

        template <typename Tag>
        itt_spinlock_init<Tag>::~itt_spinlock_init()
        {
            for (int i = 0; i < 41; ++i)
            {
                HPX_ITT_SYNC_DESTROY(&spinlock_pool<Tag>::pool_[i]);
            }
        }
    }

    template <typename Tag>
    util::detail::itt_spinlock_init<Tag>
        spinlock_pool<Tag>::init_ = util::detail::itt_spinlock_init<Tag>();
#endif

}} // namespace hpx::util

#endif
