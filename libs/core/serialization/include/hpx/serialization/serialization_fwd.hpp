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

    HPX_CORE_MODULE_EXPORT_EXTERN class access;

    HPX_CORE_MODULE_EXPORT_EXTERN struct input_archive;
    HPX_CORE_MODULE_EXPORT_EXTERN struct output_archive;

    HPX_CORE_MODULE_EXPORT_EXTERN struct binary_filter;

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    output_archive& operator<<(output_archive& ar, T const& t);

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    input_archive& operator>>(input_archive& ar, T& t);

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    output_archive& operator&(output_archive& ar, T const& t);

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    input_archive& operator&(input_archive& ar, T& t);

    namespace detail {

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        void serialize_one(output_archive& ar, T const& t);

        HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
        void serialize_one(input_archive& ar, T& t);
    }    // namespace detail

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    class array;

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename T>
    constexpr array<T> make_array(T* begin, std::size_t size) noexcept;
}    // namespace hpx::serialization
