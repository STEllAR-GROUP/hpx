//  Copyright (c) 2012-2022 Hartmut Kaiser
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
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    template <typename Tag, std::size_t N = HPX_HAVE_SPINLOCK_POOL_NUM>
    class spinlock_pool
    {
    private:
        static util::cache_aligned_data<hpx::spinlock> pool_[N];

    public:
        static hpx::spinlock& spinlock_for(void const* pv) noexcept
        {
            std::size_t i = util::fibhash<N>(reinterpret_cast<std::size_t>(pv));
            return pool_[i].data_;
        }

        class scoped_lock
        {
        private:
            hpx::spinlock& sp_;

        public:
            scoped_lock(scoped_lock const&) = delete;
            scoped_lock& operator=(scoped_lock const&) = delete;
            scoped_lock(scoped_lock&&) = delete;
            scoped_lock& operator=(scoped_lock&&) = delete;

        public:
            explicit scoped_lock(void const* pv) noexcept
              : sp_(spinlock_for(pv))
            {
                lock();
            }

            ~scoped_lock()
            {
                unlock();
            }

            void lock() noexcept(
                noexcept(std::declval<hpx::spinlock&>().lock()))
            {
                sp_.lock();
            }

            void unlock() noexcept(
                noexcept(std::declval<hpx::spinlock&>().unlock()))
            {
                sp_.unlock();
            }
        };
    };

    template <typename Tag, std::size_t N>
    util::cache_aligned_data<hpx::spinlock> spinlock_pool<Tag, N>::pool_[N];
}    // namespace hpx

namespace hpx::lcos::local {

    template <typename Tag, std::size_t N = HPX_HAVE_SPINLOCK_POOL_NUM>
    using spinlock_pool HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::spinlock_pool is deprecated, use hpx::spinlock_pool "
        "instead") = hpx::spinlock_pool<Tag, N>;
}    // namespace hpx::lcos::local
