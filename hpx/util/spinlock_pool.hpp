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
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/fibhash.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

#include <boost/smart_ptr/detail/spinlock.hpp>
#include <boost/version.hpp>

#include <cstddef>

namespace hpx { namespace util
{
    namespace detail
    {
#if HPX_HAVE_ITTNOTIFY != 0
        template <typename Tag, std::size_t N>
        struct itt_spinlock_init
        {
            itt_spinlock_init();
            ~itt_spinlock_init();
        };
#endif
#if defined(HPX_HAVE_CXX11_ALIGNAS)
        struct alignas(threads::get_cache_line_size()) spinlock_holder
        {
            boost::detail::spinlock lock;
        };
#else
        // special struct to ensure cache line alignment
        struct spinlock_holder
        {
            // pad to 64 bytes
            char cacheline_pad[threads::get_cache_line_size() -
                sizeof(boost::detail::spinlock)];
            boost::detail::spinlock lock;
        };
#endif
    }

    template <typename Tag, std::size_t N = HPX_HAVE_SPINLOCK_POOL_NUM>
    class spinlock_pool
    {
    private:
        static detail::spinlock_holder pool_[N];
#if HPX_HAVE_ITTNOTIFY != 0
        static detail::itt_spinlock_init<Tag, N> init_;
#endif

    public:

        static boost::detail::spinlock & spinlock_for( void const * pv )
        {
            std::size_t i = fibhash<N>(reinterpret_cast< std::size_t >(pv));
            return pool_[ i ].lock;
        }

        class scoped_lock
        {
        private:
            boost::detail::spinlock & sp_;

        public:
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

    template <typename Tag, std::size_t N>
    detail::spinlock_holder spinlock_pool<Tag, N>::pool_[N];

#if HPX_HAVE_ITTNOTIFY != 0
    namespace detail
    {
        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::itt_spinlock_init()
        {
            for (int i = 0; i < N; ++i)
            {
                HPX_ITT_SYNC_CREATE(&spinlock_pool<Tag, N>::pool_[i].lock,
                    "boost::detail::spinlock", 0);
                HPX_ITT_SYNC_RENAME(&spinlock_pool<Tag, N>::pool_[i].lock,
                    "boost::detail::spinlock");
            }
        }

        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::~itt_spinlock_init()
        {
            for (int i = 0; i < N; ++i)
            {
                HPX_ITT_SYNC_DESTROY(&spinlock_pool<Tag, N>::pool_[i].lock);
            }
        }
    }

    template <typename Tag, std::size_t N>
    util::detail::itt_spinlock_init<Tag, N> spinlock_pool<Tag, N>::init_;
#endif

}} // namespace hpx::util

#endif
