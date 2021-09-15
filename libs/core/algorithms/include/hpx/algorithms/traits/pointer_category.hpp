//  Copyright (c) 2016-2021 Hartmut Kaiser
//  Copyright (c) 2016 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

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

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename T, bool is_enum = std::is_enum<T>::value>
        struct unwrap_enum
        {
            using type = std::underlying_type_t<T>;
        };

        template <typename T>
        struct unwrap_enum<T, false>
        {
            using type = T;
        };

        template <typename Source, typename Dest>
        struct pointer_category_helper
        {
            using source = typename unwrap_enum<Source>::type;
            using dest = typename unwrap_enum<Dest>::type;

            using type =
                std::conditional_t<std::integral_constant<bool,
                                       sizeof(source) == sizeof(dest)>::value &&
                        std::is_integral<source>::value &&
                        std::is_integral<dest>::value &&
                        !std::is_volatile<source>::value &&
                        !std::is_volatile<dest>::value &&
                        (std::is_same<bool, source>::value ==
                            std::is_same<bool, dest>::value),
                    trivially_copyable_pointer_tag, general_pointer_tag>;
        };

        // every type is layout-compatible with itself
        template <typename T>
        struct pointer_category_helper<T, T>
        {
            using type =
                std::conditional_t<std::is_trivially_copyable<T>::value,
                    trivially_copyable_pointer_tag, general_pointer_tag>;
        };

        // pointers are layout compatible
        template <typename T>
        struct pointer_category_helper<T*, T const*>
        {
            using type = trivially_copyable_pointer_tag;
        };

        template <typename T>
        struct pointer_category_helper<T*, T volatile*>
        {
            using type = trivially_copyable_pointer_tag;
        };

        template <typename T>
        struct pointer_category_helper<T*, T const volatile*>
        {
            using type = trivially_copyable_pointer_tag;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter1, typename Iter2>
        HPX_INLINE_CONSTEXPR_VARIABLE bool iterators_are_contiguous_v =
            is_contiguous_iterator_v<Iter1>&& is_contiguous_iterator_v<Iter2>;

        template <typename Source, typename Dest,
            bool non_contiguous = !iterators_are_contiguous_v<Source, Dest>>
        struct pointer_move_category
        {
            using type = general_pointer_tag;
        };

        template <typename Source, typename Dest>
        struct pointer_move_category<Source, Dest, false>
        {
            using type = std::conditional_t<
                std::is_trivially_assignable<iter_ref_t<Dest>,
                    std::remove_reference_t<iter_ref_t<Source>>>::value,
                typename pointer_category_helper<iter_value_t<Source>,
                    iter_value_t<Dest>>::type,
                general_pointer_tag>;
        };

        template <typename Source, typename Dest,
            bool non_contiguous = !iterators_are_contiguous_v<Source, Dest>>
        struct pointer_copy_category
        {
            using type = general_pointer_tag;
        };

        template <typename Source, typename Dest>
        struct pointer_copy_category<Source, Dest, false>
        {
            using type = std::conditional_t<
                std::is_trivially_assignable<iter_ref_t<Dest>,
                    iter_ref_t<Source>>::value,
                typename pointer_category_helper<iter_value_t<Source>,
                    iter_value_t<Dest>>::type,
                general_pointer_tag>;
        };
    }    // namespace detail

    // isolate iterators that refer to contiguous trivially copyable sequences or
    // which are pointers and their value_types are assignable
    template <typename Source, typename Dest, typename Enable = void>
    struct pointer_copy_category
    {
        using type = typename detail::pointer_copy_category<Source, Dest>::type;
    };

    template <typename Source, typename Dest>
    using pointer_copy_category_t =
        typename pointer_copy_category<Source, Dest>::type;

    template <typename Source, typename Dest, typename Enable = void>
    struct pointer_move_category
    {
        using type = typename detail::pointer_move_category<Source, Dest>::type;
    };

    template <typename Source, typename Dest>
    using pointer_move_category_t =
        typename pointer_move_category<Source, Dest>::type;

    // Allow for matching of iterator<T const> to iterator<T> while calculating
    // pointer category.
    template <typename Iterator, typename Enable = void>
    struct remove_const_iterator_value_type
    {
        using type = Iterator;
    };

    template <typename Iterator>
    using remove_const_iterator_value_type_t =
        typename remove_const_iterator_value_type<Iterator>::type;
}}    // namespace hpx::traits
