//  Copyright (c) 2016-2020 Hartmut Kaiser
//  Copyright (c) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

// Select a copy tag type to enable optimization
// of copy/move operations if the iterators are
// pointers and if the value_type is layout compatible.

namespace hpx { namespace traits {

    struct general_pointer_tag
    {
    };

    struct trivially_copyable_pointer_tag : general_pointer_tag
    {
    };

    template <typename Source, typename Dest, typename Enable = void>
    struct pointer_category
    {
        using type = general_pointer_tag;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Source, typename Dest>
        struct pointer_category
        {
            using type = typename std::conditional<
                std::integral_constant<bool,
                    sizeof(Source) == sizeof(Dest)>::value &&
                    std::is_integral<Source>::value &&
                    std::is_integral<Dest>::value &&
                    !std::is_volatile<Source>::value &&
                    !std::is_volatile<Dest>::value &&
                    (std::is_same<bool, Source>::value ==
                        std::is_same<bool, Dest>::value),
                trivially_copyable_pointer_tag, general_pointer_tag>::type;
        };

        // every type is layout-compatible with itself
        template <typename T>
        struct pointer_category<T, T>
        {
            using type =
                typename std::conditional<std::is_trivially_copyable<T>::value,
                    trivially_copyable_pointer_tag, general_pointer_tag>::type;
        };

        // pointers are layout compatible
        template <typename T>
        struct pointer_category<T const*, T*>
        {
            using type = trivially_copyable_pointer_tag;
        };
    }    // namespace detail

    // isolate iterators which are pointers and their value_types are
    // assignable
    template <typename Source, typename Dest>
    struct pointer_category<Source*, Dest*>
    {
        using type =
            typename std::conditional<std::is_assignable<Dest&, Source&>::value,
                typename detail::pointer_category<
                    typename std::remove_const<Source>::type, Dest>::type,
                general_pointer_tag>::type;
    };

    // Allow for matching of iterator<T const> to iterator<T> while calculating
    // pointer category.
    template <typename Iterator, typename Enable = void>
    struct remove_const_iterator_value_type
    {
        using type = Iterator;
    };
}}    // namespace hpx::traits
