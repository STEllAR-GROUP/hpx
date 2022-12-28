// (C) Copyright 2007 Anthony Williams
// (C) Copyright 2011-2012 Vicente J. Botet Escriba
// Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>

#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace hpx {

    template <typename Mutex>
    class upgrade_to_unique_lock;

    template <typename Mutex>
    class upgrade_lock
    {
    protected:
        Mutex* m;
        bool is_locked;

        template <typename Mutex_>
        friend class upgrade_to_unique_lock;

    public:
        using mutex_type = Mutex;

        upgrade_lock(upgrade_lock const&) = delete;
        upgrade_lock& operator=(upgrade_lock const&) = delete;

        upgrade_lock() noexcept
          : m(nullptr)
          , is_locked(false)
        {
        }

        explicit upgrade_lock(Mutex& m_)
          : m(&m_)
          , is_locked(false)
        {
            lock();
        }

        upgrade_lock(Mutex& m_, std::adopt_lock_t)
          : m(&m_)
          , is_locked(true)
        {
        }

        upgrade_lock(Mutex& m_, std::defer_lock_t) noexcept
          : m(&m_)
          , is_locked(false)
        {
        }

        upgrade_lock(Mutex& m_, std::try_to_lock_t)
          : m(&m_)
          , is_locked(false)
        {
            try_lock();
        }

        upgrade_lock(upgrade_lock<Mutex>&& other) noexcept
          : m(other.m)
          , is_locked(other.is_locked)
        {
            other.is_locked = false;
            other.m = nullptr;
        }

        explicit upgrade_lock(std::unique_lock<Mutex>&& other)
          : m(other.mutex())
          , is_locked(other.owns_lock())
        {
            if (is_locked)
            {
                m->unlock_and_lock_upgrade();
            }
            other.release();
        }

        upgrade_lock& operator=(upgrade_lock<Mutex>&& other) noexcept
        {
            upgrade_lock temp(HPX_MOVE(other));
            swap(temp);
            return *this;
        }

        void swap(upgrade_lock& other) noexcept
        {
            std::swap(m, other.m);
            std::swap(is_locked, other.is_locked);
        }

        Mutex* mutex() const noexcept
        {
            return m;
        }

        Mutex* release() noexcept
        {
            Mutex* const res = m;
            m = nullptr;
            is_locked = false;
            return res;
        }

        ~upgrade_lock()
        {
            if (owns_lock())
            {
                m->unlock_upgrade();
            }
        }

        void lock()
        {
            if (m == nullptr)
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock has no mutex");
            }
            if (owns_lock())
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock already owns the mutex");
            }
            m->lock_upgrade();
            is_locked = true;
        }

        bool try_lock()
        {
            if (m == nullptr)
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock has no mutex");
            }
            if (owns_lock())
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock already owns the mutex");
            }
            is_locked = m->try_lock_upgrade();
            return is_locked;
        }

        void unlock()
        {
            if (m == nullptr)
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock has no mutex");
            }
            if (!owns_lock())
            {
                HPX_THROW_EXCEPTION(hpx::error::lock_error, "mutex::unlock",
                    "upgrade_lock doesn't own the mutex");
            }
            m->unlock_upgrade();
            is_locked = false;
        }

        explicit constexpr operator bool() const noexcept
        {
            return owns_lock();
        }

        constexpr bool owns_lock() const noexcept
        {
            return is_locked;
        }
    };

    template <typename Mutex>
    void swap(upgrade_lock<Mutex>& lhs, upgrade_lock<Mutex>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    template <typename Mutex>
    class upgrade_to_unique_lock
    {
    private:
        upgrade_lock<Mutex>* source;
        std::unique_lock<Mutex> exclusive;

    public:
        using mutex_type = Mutex;

        upgrade_to_unique_lock(upgrade_to_unique_lock const&) = delete;
        upgrade_to_unique_lock& operator=(
            upgrade_to_unique_lock const&) = delete;

        explicit upgrade_to_unique_lock(upgrade_lock<Mutex>& m_)
          : source(&m_)
          , exclusive()
        {
            if (m_.is_locked)
            {
                exclusive = std::unique_lock<Mutex>(*m_.m, std::adopt_lock);
                m_.m->unlock_upgrade_and_lock();
            }
            else
            {
                exclusive = std::unique_lock<Mutex>(*m_.m, std::defer_lock);
            }
            m_.release();
        }

        ~upgrade_to_unique_lock()
        {
            if (source != nullptr)
            {
                *source = upgrade_lock<Mutex>(HPX_MOVE(exclusive));
            }
        }

        upgrade_to_unique_lock(upgrade_to_unique_lock<Mutex>&& other) noexcept
          : source(other.source)
          , exclusive(HPX_MOVE(other.exclusive))
        {
            other.source = nullptr;
        }

        upgrade_to_unique_lock& operator=(
            upgrade_to_unique_lock<Mutex>&& other) noexcept
        {
            upgrade_to_unique_lock temp(HPX_MOVE(other));
            swap(temp);
            return *this;
        }

        void swap(upgrade_to_unique_lock& other) noexcept
        {
            std::swap(source, other.source);
            exclusive.swap(other.exclusive);
        }

        explicit constexpr operator bool() const noexcept
        {
            return owns_lock();
        }

        constexpr bool owns_lock() const noexcept
        {
            return exclusive.owns_lock();
        }

        Mutex* mutex() const noexcept
        {
            return exclusive.mutex();
        }
    };
}    // namespace hpx

namespace hpx::lcos::local {

    template <typename Mutex>
    using upgrade_to_unique_lock HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::upgrade_to_unique_lock is deprecated, use "
        "hpx::upgrade_to_unique_lock instead") =
        hpx::upgrade_to_unique_lock<Mutex>;

    template <typename Mutex>
    using upgrade_lock HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::local::upgrade_lock is deprecated, use hpx::upgrade_lock "
        "instead") = hpx::upgrade_lock<Mutex>;
}    // namespace hpx::lcos::local
