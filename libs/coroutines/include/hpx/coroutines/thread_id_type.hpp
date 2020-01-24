//  Copyright (c) 2018 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/coroutines/thread_id_type.hpp

#ifndef HPX_THREADS_THREAD_ID_TYPE_HPP
#define HPX_THREADS_THREAD_ID_TYPE_HPP

#include <hpx/config/constexpr.hpp>
#include <hpx/config/export_definitions.hpp>

#include <cstddef>
#include <functional>
#include <iosfwd>

namespace hpx { namespace threads {

    struct thread_id
    {
    private:
        using thread_id_repr = void*;

    public:
        constexpr thread_id() noexcept
          : thrd_(nullptr)
        {
        }
        explicit constexpr thread_id(thread_id_repr thrd) noexcept
          : thrd_(thrd)
        {
        }

        thread_id(thread_id const&) = default;
        thread_id& operator=(thread_id const&) = default;

        thread_id(thread_id&& rhs) noexcept
          : thrd_(rhs.thrd_)
        {
            rhs.thrd_ = nullptr;
        }

        thread_id& operator=(thread_id&& rhs) noexcept
        {
            if (&rhs != this)
            {
                thrd_ = rhs.thrd_;
                rhs.thrd_ = nullptr;
            }
            return *this;
        }

        explicit constexpr operator bool() const noexcept
        {
            return nullptr != thrd_;
        }

        constexpr thread_id_repr get() const noexcept
        {
            return thrd_;
        }

        HPX_CXX14_CONSTEXPR void reset() noexcept
        {
            thrd_ = nullptr;
        }

        friend constexpr bool operator==(
            std::nullptr_t, thread_id const& rhs) noexcept
        {
            return nullptr == rhs.thrd_;
        }

        friend constexpr bool operator!=(
            std::nullptr_t, thread_id const& rhs) noexcept
        {
            return nullptr != rhs.thrd_;
        }

        friend constexpr bool operator==(
            thread_id const& lhs, std::nullptr_t) noexcept
        {
            return nullptr == lhs.thrd_;
        }

        friend constexpr bool operator!=(
            thread_id const& lhs, std::nullptr_t) noexcept
        {
            return nullptr != lhs.thrd_;
        }

        friend constexpr bool operator==(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return lhs.thrd_ == rhs.thrd_;
        }

        friend constexpr bool operator!=(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return lhs.thrd_ != rhs.thrd_;
        }

        friend HPX_CXX14_CONSTEXPR bool operator<(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return std::less<void const*>{}(lhs.thrd_, rhs.thrd_);
        }

        friend HPX_CXX14_CONSTEXPR bool operator>(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return std::less<void const*>{}(rhs.thrd_, lhs.thrd_);
        }

        friend HPX_CXX14_CONSTEXPR bool operator<=(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return !(rhs > lhs);
        }

        friend HPX_CXX14_CONSTEXPR bool operator>=(
            thread_id const& lhs, thread_id const& rhs) noexcept
        {
            return !(rhs < lhs);
        }

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>& operator<<(
            std::basic_ostream<Char, Traits>& os, thread_id const& id)
        {
            os << id.get();
            return os;
        }

    private:
        thread_id_repr thrd_;
    };

    HPX_CONSTEXPR_OR_CONST thread_id invalid_thread_id;

}}    // namespace hpx::threads

#endif
