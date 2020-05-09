//  Copyright (c) 2012 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  adapted from:
//  boost/detail/spinlock_pool.hpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/hashing/fibhash.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    namespace detail {
#if HPX_HAVE_ITTNOTIFY != 0
        template <typename Tag, std::size_t N>
        struct itt_spinlock_init
        {
            itt_spinlock_init();
            ~itt_spinlock_init();
        };
#endif
    }    // namespace detail

    template <typename Tag, std::size_t N = HPX_HAVE_SPINLOCK_POOL_NUM>
    class spinlock_pool
    {
    private:
        static util::cache_aligned_data<lcos::local::spinlock> pool_[N];
#if HPX_HAVE_ITTNOTIFY != 0
        static detail::itt_spinlock_init<Tag> init_;
#endif
    public:
        static lcos::local::spinlock& spinlock_for(void const* pv)
        {
            std::size_t i = util::fibhash<N>(reinterpret_cast<std::size_t>(pv));
            return pool_[i].data_;
        }

        class scoped_lock
        {
        private:
            hpx::lcos::local::spinlock& sp_;

        public:
            HPX_NON_COPYABLE(scoped_lock);

        public:
            explicit scoped_lock(void const* pv)
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
            }

            void unlock()
            {
                HPX_ITT_SYNC_RELEASING(&sp_);
                sp_.unlock();
                HPX_ITT_SYNC_RELEASED(&sp_);
            }
        };
    };

    template <typename Tag, std::size_t N>
    util::cache_aligned_data<lcos::local::spinlock>
        spinlock_pool<Tag, N>::pool_[N];

#if HPX_HAVE_ITTNOTIFY != 0
    namespace detail {
        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::itt_spinlock_init()
        {
            for (int i = 0; i < 41; ++i)
            {
                HPX_ITT_SYNC_CREATE(
                    &lcos::local::spinlock_pool<Tag, N>::pool_[i].data_,
                    "hpx::lcos::spinlock", 0);
                HPX_ITT_SYNC_RENAME(
                    &lcos::local::spinlock_pool<Tag, N>::pool_[i].data_,
                    "hpx::lcos::spinlock");
            }
        }

        template <typename Tag, std::size_t N>
        itt_spinlock_init<Tag, N>::~itt_spinlock_init()
        {
            for (int i = 0; i < N; ++i)
            {
                HPX_ITT_SYNC_DESTROY(&spinlock_pool<Tag, N>::pool_[i].data_);
            }
        }
    }    // namespace detail

    template <typename Tag, std::size_t N>
    util::detail::itt_spinlock_init<Tag, N> spinlock_pool<Tag, N>::init_;
#endif
}}}    // namespace hpx::lcos::local
