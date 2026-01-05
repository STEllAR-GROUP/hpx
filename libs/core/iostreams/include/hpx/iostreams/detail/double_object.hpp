//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2004-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

// Contains the definition of the class template
// hpx::iostreams::detail::double_object, which is similar to compressed pair
// except that both members of the pair have the same type, and compression
// occurs only if requested using a boolean template parameter.

#pragma once

#include <hpx/config.hpp>

#include <algorithm>
#include <type_traits>

namespace hpx::iostreams::detail {

    template <typename T>
    class single_object_holder
    {
    public:
        single_object_holder() = default;

        template <typename T_>
            requires(std::is_convertible_v<std::decay_t<T_>, T>)
        explicit single_object_holder(T_&& t, T_&&)
          : first_(HPX_FORWARD(T_, t))
        {
        }

        T& first() noexcept
        {
            return first_;
        }

        T const& first() const noexcept
        {
            return first_;
        }

        T& second() noexcept
        {
            return first_;
        }

        T const& second() const noexcept
        {
            return first_;
        }

        void swap(single_object_holder& o) noexcept
        {
            using std::swap;
            swap(first_, o.first_);
        }

    private:
        T first_ = T();
    };

    template <typename T>
    struct double_object_holder
    {
        double_object_holder() = default;

        template <typename T_>
            requires(std::is_convertible_v<std::decay_t<T_>, T>)
        double_object_holder(T_&& t1, T_&& t2)
          : first_(HPX_FORWARD(T_, t1))
          , second_(HPX_FORWARD(T_, t2))
        {
        }

        T& first() noexcept
        {
            return first_;
        }

        T const& first() const noexcept
        {
            return first_;
        }

        T& second() noexcept
        {
            return second_;
        }

        T const& second() const noexcept
        {
            return second_;
        }

        void swap(double_object_holder& d) noexcept
        {
            using std::swap;
            swap(first_, d.first_);
            swap(second_, d.second_);
        }

    private:
        T first_ = T(), second_ = T();
    };

    template <typename T, typename IsDouble>
    class double_object
      : public std::conditional_t<IsDouble::value, double_object_holder<T>,
            single_object_holder<T>>
    {
        using base_type = std::conditional_t<IsDouble::value,
            double_object_holder<T>, single_object_holder<T>>;

    public:
        double_object() = default;

        template <typename T_>
            requires(std::is_convertible_v<std::decay_t<T_>, T>)
        double_object(T_&& t1, T_&& t2)
          : base_type(HPX_FORWARD(T_, t1), HPX_FORWARD(T_, t2))
        {
        }

        [[nodiscard]] static constexpr bool is_double() noexcept
        {
            return IsDouble::value;
        }
    };
}    // namespace hpx::iostreams::detail
