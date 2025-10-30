//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/modules/preprocessor.hpp>
#include <hpx/serialization/macros.hpp>

#include <cstddef>

namespace hpx::serialization {

    HPX_CXX_EXPORT HPX_CXX_EXTERN class access;

    HPX_CXX_EXPORT HPX_CXX_EXTERN struct input_archive;
    HPX_CXX_EXPORT HPX_CXX_EXTERN struct output_archive;

    HPX_CXX_EXPORT struct binary_filter;

    HPX_CXX_EXPORT template <typename T>
    output_archive& operator<<(output_archive& ar, T const& t);

    HPX_CXX_EXPORT template <typename T>
    input_archive& operator>>(input_archive& ar, T& t);

    HPX_CXX_EXPORT template <typename T>
    output_archive& operator&(output_archive& ar, T const& t);

    HPX_CXX_EXPORT template <typename T>
    input_archive& operator&(input_archive& ar, T& t);

    namespace detail {

        HPX_CXX_EXPORT template <typename T>
        void serialize_one(output_archive& ar, T const& t);

        HPX_CXX_EXPORT template <typename T>
        void serialize_one(input_archive& ar, T& t);
    }    // namespace detail

    HPX_CXX_EXPORT template <typename T>
    class array;

    HPX_CXX_EXPORT template <typename T>
    constexpr array<T> make_array(T* begin, std::size_t size) noexcept;
}    // namespace hpx::serialization
