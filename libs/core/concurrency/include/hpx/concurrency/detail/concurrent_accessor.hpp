//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/type_support/assert_owns_lock.hpp>

#include <concepts>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx::concurrent::detail {

    // Generic accessor for thread-safe access to container elements
    template <typename T>
    class concurrent_accessor
    {
    public:
        using value_type = T;
        using reference_type = T&;
        using pointer_type = T*;

    private:
        std::unique_lock<hpx::util::spinlock> lock_;
        pointer_type value_ = nullptr;

        void validate() const
        {
            if (!value_)
            {
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "concurrent_accessor", "Empty accessor dereference");
            }
        }

    public:
        concurrent_accessor() = default;

        concurrent_accessor(
            std::unique_lock<hpx::util::spinlock>&& l, reference_type v)
          : lock_(HPX_MOVE(l))
          , value_(&v)
        {
            HPX_ASSERT_OWNS_LOCK(lock_);
        }

        bool empty() const noexcept
        {
            return value_ == nullptr;
        }

        explicit operator bool() const noexcept
            requires(!std::same_as<std::remove_const_t<T>, bool>)
        {
            return !empty();
        }

        operator reference_type() const
        {
            return get();
        }

        reference_type get() const
        {
            validate();
            return *value_;
        }

        // Assignment operator only for non-const accessor
        concurrent_accessor& operator=(T const& v)
            requires(!std::is_const_v<T>)
        {
            validate();
            *value_ = v;
            return *this;
        }
    };

}    // namespace hpx::concurrent::detail
