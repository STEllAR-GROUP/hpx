#ifndef HPX_UTIL_SPINLOCK_POOL_HPP_INCLUDED
#define HPX_UTIL_SPINLOCK_POOL_HPP_INCLUDED

// MS compatible compilers support #pragma once

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

//
//  taken from:
//  boost/detail/spinlock_pool.hpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  spinlock_pool<0> is reserved for atomic<>, when/if it arrives
//  spinlock_pool<1> is reserved for shared_ptr reference counts
//  spinlock_pool<2> is reserved for shared_ptr atomic access
//

#include <boost/config.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 103900
#include <boost/smart_ptr/detail/spinlock.hpp>
#else
#include <boost/detail/spinlock.hpp>
#endif
#include <cstddef>

#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

namespace hpx { namespace util
{
#if HPX_USE_ITTNOTIFY != 0
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
#if HPX_USE_ITTNOTIFY != 0
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

            scoped_lock( scoped_lock const & );
            scoped_lock & operator=( scoped_lock const & );

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

#if HPX_USE_ITTNOTIFY != 0
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
