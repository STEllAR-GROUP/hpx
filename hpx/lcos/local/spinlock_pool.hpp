//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  adapted from:
//  boost/detail/spinlock_pool.hpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_SPINLOCK_POOL_HPP
#define HPX_LCOS_LOCAL_SPINLOCK_POOL_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
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
        static const std::size_t N = 41;
        static lcos::local::spinlock pool_[N];
#if HPX_HAVE_ITTNOTIFY != 0
        static detail::itt_spinlock_init<Tag> init_;
#endif
    public:
        static lcos::local::spinlock & spinlock_for(void const * pv)
        {
            std::size_t i = reinterpret_cast< std::size_t >( pv ) % N;
            return pool_[ i ];
        }

        class scoped_lock
        {
        private:
            hpx::lcos::local::spinlock & sp_;

            HPX_NON_COPYABLE(scoped_lock);

        public:
            explicit scoped_lock(void const * pv)
              : sp_(spinlock_for(pv))
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
    lcos::local::spinlock spinlock_pool<Tag>::pool_[spinlock_pool<Tag>::N];

#if HPX_HAVE_ITTNOTIFY != 0
    namespace detail
    {
        template <typename Tag>
        itt_spinlock_init<Tag>::itt_spinlock_init()
        {
            for (int i = 0; i < 41; ++i)
            {
                HPX_ITT_SYNC_CREATE(&lcos::local::spinlock_pool<Tag>::pool_[i],
                    "hpx::lcos::spinlock", 0);
                HPX_ITT_SYNC_RENAME(&lcos::local::spinlock_pool<Tag>::pool_[i],
                    "hpx::lcos::spinlock");
            }
        }

        template <typename Tag>
        itt_spinlock_init<Tag>::~itt_spinlock_init()
        {
            for (int i = 0; i < 41; ++i)
            {
                HPX_ITT_SYNC_DESTROY(&lcos::local::spinlock_pool<Tag>::pool_[i]);
            }
        }
    }

    template <typename Tag>
    lcos::local::detail::itt_spinlock_init<Tag>
        lcos::local::spinlock_pool<Tag>::init_ =
            lcos::local::detail::itt_spinlock_init<Tag>();
#endif
}}}

#endif
