//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/preprocessor/strip_parens.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::serialization {

    class access;

    struct input_archive;
    struct output_archive;

    struct binary_filter;

    template <typename T>
    output_archive& operator<<(output_archive& ar, T const& t);

    template <typename T>
    input_archive& operator>>(input_archive& ar, T& t);

    template <typename T>
    output_archive& operator&(output_archive& ar, T const& t);

    template <typename T>
    input_archive& operator&(input_archive& ar, T& t);

    namespace detail {

        template <typename Archive, typename T>
        void serialize_one(Archive& ar, T& t);
    }

    template <typename T>
    class array;

    template <typename T>
    constexpr array<T> make_array(T* begin, std::size_t size) noexcept;
}    // namespace hpx::serialization

#define HPX_SERIALIZATION_SPLIT_MEMBER()                                       \
    void serialize(hpx::serialization::input_archive& ar, unsigned)            \
    {                                                                          \
        load(ar, 0);                                                           \
    }                                                                          \
    void serialize(hpx::serialization::output_archive& ar, unsigned) const     \
    {                                                                          \
        save(ar, 0);                                                           \
    }                                                                          \
    /**/

#define HPX_SERIALIZATION_SPLIT_FREE(T)                                        \
    HPX_FORCEINLINE void serialize(                                            \
        hpx::serialization::input_archive& ar, T& t, unsigned)                 \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    HPX_FORCEINLINE void serialize(                                            \
        hpx::serialization::output_archive& ar, T& t, unsigned)                \
    {                                                                          \
        save(ar, const_cast<std::add_const_t<T>&>(t), 0);                      \
    }                                                                          \
    /**/

#define HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(TEMPLATE, ARGS)                  \
    HPX_PP_STRIP_PARENS(TEMPLATE)                                              \
    HPX_FORCEINLINE void serialize(hpx::serialization::input_archive& ar,      \
        HPX_PP_STRIP_PARENS(ARGS) & t, unsigned)                               \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    HPX_PP_STRIP_PARENS(TEMPLATE)                                              \
    HPX_FORCEINLINE void serialize(hpx::serialization::output_archive& ar,     \
        HPX_PP_STRIP_PARENS(ARGS) & t, unsigned)                               \
    {                                                                          \
        save(ar, const_cast<std::add_const_t<HPX_PP_STRIP_PARENS(ARGS)>&>(t),  \
            0);                                                                \
    }                                                                          \
    /**/
