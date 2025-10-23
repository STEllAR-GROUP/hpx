//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2019 Austin McCartney
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/boost_iterator_categories.hpp>
#include <hpx/modules/type_support.hpp>

#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter>
    using iter_value_t = typename std::iterator_traits<Iter>::value_type;

    HPX_CXX_EXPORT template <typename Iter>
    using iter_difference_t =
        typename std::iterator_traits<Iter>::difference_type;

    HPX_CXX_EXPORT template <typename Iter>
    using iter_pointer_t = typename std::iterator_traits<Iter>::pointer;

    HPX_CXX_EXPORT template <typename Iter>
    using iter_reference_t = typename std::iterator_traits<Iter>::reference;

    HPX_CXX_EXPORT template <typename Iter>
    using iter_category_t =
        typename std::iterator_traits<Iter>::iterator_category;

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        // This implementation of is_iterator seems to work fine even for VS2013
        // which has an implementation of std::iterator_traits that is
        // SFINAE-unfriendly.
        HPX_CXX_EXPORT template <typename T>
        struct is_iterator
        {
#if defined(HPX_MSVC) && defined(__CUDACC__)
            template <typename U>
            static typename U::iterator_category* test(U);    // iterator

            template <typename U>
            static void* test(U*);    // pointer
#else
            template <typename U, typename = iter_pointer_t<U>>
            static void* test(U&&);
#endif

            static char test(...);

            static constexpr bool value =
                sizeof(test(std::declval<T>())) == sizeof(void*);
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename U, typename Enable = void>
        struct addition_result
        {
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        struct addition_result<T, U,
            std::void_t<decltype(std::declval<T>() + std::declval<U>())>>
        {
            using type = decltype(std::declval<T>() + std::declval<U>());
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        using addition_result_t = typename addition_result<T, U>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename Enable = void>
        struct dereference_result
        {
        };

        HPX_CXX_EXPORT template <typename T>
        struct dereference_result<T,
            std::void_t<decltype(*(std::declval<T&>()))>>
        {
            using type = decltype(*(std::declval<T&>()));
        };

        HPX_CXX_EXPORT template <typename T>
        using dereference_result_t = typename dereference_result<T>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename U, typename Enable = void>
        struct inplace_addition_result
        {
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        struct inplace_addition_result<T, U,
            std::void_t<decltype(std::declval<T>() += std::declval<U>())>>
        {
            using type = decltype(std::declval<T>() += std::declval<U>());
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        using inplace_addition_result_t =
            typename inplace_addition_result<T, U>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename U, typename Enable = void>
        struct inplace_subtraction_result
        {
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        struct inplace_subtraction_result<T, U,
            std::void_t<decltype(std::declval<T>() -= std::declval<U>())>>
        {
            using type = decltype(std::declval<T>() -= std::declval<U>());
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        using inplace_subtraction_result_t =
            typename inplace_subtraction_result<T, U>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename Enable = void>
        struct predecrement_result
        {
        };

        HPX_CXX_EXPORT template <typename T>
        struct predecrement_result<T,
            std::void_t<decltype(--std::declval<T&>())>>
        {
            using type = decltype(--std::declval<T&>());
        };

        HPX_CXX_EXPORT template <typename T>
        using predecrement_result_t = typename predecrement_result<T>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename Enable = void>
        struct preincrement_result
        {
        };

        HPX_CXX_EXPORT template <typename T>
        struct preincrement_result<T,
            std::void_t<decltype(++std::declval<T&>())>>
        {
            using type = decltype(++std::declval<T&>());
        };

        HPX_CXX_EXPORT template <typename T>
        using preincrement_result_t = typename preincrement_result<T>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename Enable = void>
        struct postdecrement_result
        {
        };

        HPX_CXX_EXPORT template <typename T>
        struct postdecrement_result<T,
            std::void_t<decltype(std::declval<T&>()--)>>
        {
            using type = decltype(std::declval<T&>()--);
        };

        HPX_CXX_EXPORT template <typename T>
        using postdecrement_result_t = typename postdecrement_result<T>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename Enable = void>
        struct postincrement_result
        {
        };

        HPX_CXX_EXPORT template <typename T>
        struct postincrement_result<T,
            std::void_t<decltype(std::declval<T&>()++)>>
        {
            using type = decltype(std::declval<T&>()++);
        };

        HPX_CXX_EXPORT template <typename T>
        using postincrement_result_t = typename postincrement_result<T>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename U, typename Enable = void>
        struct subscript_result
        {
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        struct subscript_result<T, U,
            std::void_t<decltype(std::declval<T&>()[std::declval<U>()])>>
        {
            using type = decltype(std::declval<T&>()[std::declval<U>()]);
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        using subscript_result_t = typename subscript_result<T, U>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename T, typename U, typename Enable = void>
        struct subtraction_result
        {
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        struct subtraction_result<T, U,
            std::void_t<decltype(std::declval<T>() - std::declval<U>())>>
        {
            using type = decltype(std::declval<T>() - std::declval<U>());
        };

        HPX_CXX_EXPORT template <typename T, typename U>
        using subtraction_result_t = typename subtraction_result<T, U>::type;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
        struct bidirectional_concept : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct bidirectional_concept<Iter,
            std::void_t<dereference_result_t<Iter>,
                equality_result_t<Iter, Iter>, inequality_result_t<Iter, Iter>,
                predecrement_result_t<Iter>, preincrement_result_t<Iter>,
                postdecrement_result_t<Iter>, postincrement_result_t<Iter>>>
          : std::integral_constant<bool,
                std::is_convertible_v<bool, equality_result_t<Iter, Iter>> &&
                    std::is_convertible_v<bool,
                        inequality_result_t<Iter, Iter>> &&
                    std::is_same_v<std::add_lvalue_reference_t<Iter>,
                        predecrement_result_t<Iter>> &&
                    std::is_same_v<std::add_lvalue_reference_t<Iter>,
                        preincrement_result_t<Iter>> &&
                    std::is_same_v<Iter, postdecrement_result_t<Iter>> &&
                    std::is_same_v<Iter, postincrement_result_t<Iter>>>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
        struct random_access_concept : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct random_access_concept<Iter,
            std::void_t<dereference_result_t<Iter>,
                subscript_result_t<Iter, iter_difference_t<Iter>>,
                addition_result_t<Iter, iter_difference_t<Iter>>,
                inplace_addition_result_t<Iter, iter_difference_t<Iter>>,
                subtraction_result_t<Iter, iter_difference_t<Iter>>,
                subtraction_result_t<Iter, Iter>,
                inplace_subtraction_result_t<Iter, iter_difference_t<Iter>>>>
          : std::integral_constant<bool,
                bidirectional_concept<Iter>::value &&
                    std::is_same_v<dereference_result_t<Iter>,
                        subscript_result_t<Iter, iter_difference_t<Iter>>> &&
                    std::is_same_v<Iter,
                        addition_result_t<Iter, iter_difference_t<Iter>>> &&
                    std::is_same_v<std::add_lvalue_reference_t<Iter>,
                        inplace_addition_result_t<Iter,
                            iter_difference_t<Iter>>> &&
                    std::is_same_v<Iter,
                        subtraction_result_t<Iter, iter_difference_t<Iter>>> &&
                    std::is_same_v<iter_difference_t<Iter>,
                        subtraction_result_t<Iter, Iter>> &&
                    std::is_same_v<std::add_lvalue_reference_t<Iter>,
                        inplace_subtraction_result_t<Iter,
                            iter_difference_t<Iter>>>>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename TraversalTag>
        struct satisfy_traversal_concept : std::false_type
        {
        };

        // The interface guarantees of InputIterator and ForwardIterator
        // concepts are not sufficient to robustly distinguish whether a given
        // type models a forward iterator or an input iterator through interface
        // inspection and concept emulation alone, given the type in question
        // models an iterator category no stronger than forward.
        //
        // That said, a type which models the BidirectionalIterator concept also
        // models a ForwardIterator concept, by definition (and the interface
        // guarantees on the BidirectionalIterator concept are sufficient for
        // robust concept checking). Here we provide a specialization to capture
        // this case, such that, bidirectional and random access iterators will
        // be properly recognized as satisfying the ForwardIterator concept.
        HPX_CXX_EXPORT template <typename Iter>
        struct satisfy_traversal_concept<Iter, hpx::forward_traversal_tag>
          : bidirectional_concept<Iter>
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct satisfy_traversal_concept<Iter, hpx::bidirectional_traversal_tag>
          : bidirectional_concept<Iter>
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct satisfy_traversal_concept<Iter, hpx::random_access_traversal_tag>
          : random_access_concept<Iter>
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename TraversalTag>
        inline constexpr bool satisfy_traversal_concept_v =
            satisfy_traversal_concept<Iter, TraversalTag>::value;
    }    // namespace detail

    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_iterator : detail::is_iterator<std::decay_t<Iter>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_iterator_t = typename is_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_iterator_v = is_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Cat,
            typename Enable = void>
        struct belongs_to_iterator_category : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Cat>
        struct belongs_to_iterator_category<Iter, Cat,
            std::enable_if_t<is_iterator_v<Iter>>>
          : std::is_base_of<Cat, iter_category_t<Iter>>
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Cat>
        inline constexpr bool belongs_to_iterator_category_v =
            belongs_to_iterator_category<Iter, Cat>::value;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Traversal,
            typename Enable = void>
        struct belongs_to_iterator_traversal : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Traversal>
        struct belongs_to_iterator_traversal<Iter, Traversal,
            std::enable_if_t<is_iterator_v<Iter>>>
          : std::integral_constant<bool,
                std::is_base_of_v<Traversal,
                    hpx::traits::iterator_traversal_t<Iter>> ||
                    satisfy_traversal_concept_v<Iter, Traversal>>
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Traversal>
        inline constexpr bool belongs_to_iterator_traversal_v =
            belongs_to_iterator_traversal<Iter, Traversal>::value;

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Cat,
            typename Enable = void>
        struct has_category : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Cat>
        struct has_category<Iter, Cat, std::enable_if_t<is_iterator_v<Iter>>>
          : std::is_same<Cat, iter_category_t<Iter>>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT std::random_access_iterator_tag coerce_iterator_tag(
            std::random_access_iterator_tag const&);
        HPX_CXX_EXPORT std::bidirectional_iterator_tag coerce_iterator_tag(
            std::bidirectional_iterator_tag const&);
        HPX_CXX_EXPORT std::forward_iterator_tag coerce_iterator_tag(
            std::forward_iterator_tag const&);
        HPX_CXX_EXPORT std::input_iterator_tag coerce_iterator_tag(
            std::input_iterator_tag const&);
        HPX_CXX_EXPORT std::output_iterator_tag coerce_iterator_tag(
            std::output_iterator_tag const&);

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_EXPORT template <typename Iter, typename Traversal,
            typename Enable = void>
        struct has_traversal : std::false_type
        {
        };

        HPX_CXX_EXPORT template <typename Iter, typename Traversal>
        struct has_traversal<Iter, Traversal,
            std::enable_if_t<is_iterator_v<Iter>>>
          : std::is_same<Traversal, hpx::traits::iterator_traversal_t<Iter>>
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct has_traversal<Iter, hpx::bidirectional_traversal_tag,
            std::enable_if_t<is_iterator_v<Iter>>>
          : std::integral_constant<bool,
                std::is_same_v<hpx::bidirectional_traversal_tag,
                    hpx::traits::iterator_traversal_t<Iter>> ||
                    (satisfy_traversal_concept_v<Iter,
                         hpx::bidirectional_traversal_tag> &&
                        !satisfy_traversal_concept_v<Iter,
                            hpx::random_access_traversal_tag>)>
        {
        };

        HPX_CXX_EXPORT template <typename Iter>
        struct has_traversal<Iter, hpx::random_access_traversal_tag,
            std::enable_if_t<is_iterator_v<Iter>>>
          : std::integral_constant<bool,
                std::is_same_v<hpx::random_access_traversal_tag,
                    hpx::traits::iterator_traversal_t<Iter>> ||
                    satisfy_traversal_concept_v<Iter,
                        hpx::random_access_traversal_tag>>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Category>
    inline constexpr bool has_category_v =
        detail::has_category<std::decay_t<Iter>, Category>::value;

    HPX_CXX_EXPORT template <typename Iter, typename Traversal>
    inline constexpr bool has_traversal_v =
        detail::has_traversal<std::decay_t<Iter>, Traversal>::value;

    HPX_CXX_EXPORT template <typename Iter, typename Category>
    inline constexpr bool belongs_to_iterator_category_v =
        detail::belongs_to_iterator_category<std::decay_t<Iter>,
            Category>::value;

    HPX_CXX_EXPORT template <typename Iter, typename Traversal>
    inline constexpr bool belongs_to_iterator_traversal_v =
        detail::belongs_to_iterator_traversal<std::decay_t<Iter>,
            Traversal>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_output_iterator
      : std::integral_constant<bool,
            belongs_to_iterator_category_v<Iter, std::output_iterator_tag> ||
                belongs_to_iterator_traversal_v<Iter,
                    hpx::incrementable_traversal_tag>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_output_iterator_t = typename is_output_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_output_iterator_v =
        is_output_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_input_iterator
      : std::integral_constant<bool,
            belongs_to_iterator_category_v<Iter, std::input_iterator_tag> ||
                belongs_to_iterator_traversal_v<Iter,
                    hpx::single_pass_traversal_tag>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_input_iterator_t = typename is_input_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_input_iterator_v = is_input_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_forward_iterator
      : std::integral_constant<bool,
            belongs_to_iterator_category_v<Iter, std::forward_iterator_tag> ||
                belongs_to_iterator_traversal_v<Iter,
                    hpx::forward_traversal_tag>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_forward_iterator_t = typename is_forward_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_forward_iterator_v =
        is_forward_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_bidirectional_iterator
      : std::integral_constant<bool,
            belongs_to_iterator_category_v<Iter,
                std::bidirectional_iterator_tag> ||
                belongs_to_iterator_traversal_v<Iter,
                    hpx::bidirectional_traversal_tag>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_bidirectional_iterator_t =
        typename is_bidirectional_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_bidirectional_iterator_v =
        is_bidirectional_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_random_access_iterator
      : std::integral_constant<bool,
            has_category_v<Iter, std::random_access_iterator_tag> ||
                has_traversal_v<Iter, hpx::random_access_traversal_tag>>
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_random_access_iterator_t =
        typename is_random_access_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_random_access_iterator_v =
        is_random_access_iterator<Iter>::value;

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename Iter, typename Enable = void>
    struct is_zip_iterator : std::false_type
    {
    };

    HPX_CXX_EXPORT template <typename Iter>
    using is_zip_iterator_t = typename is_zip_iterator<Iter>::type;

    HPX_CXX_EXPORT template <typename Iter>
    inline constexpr bool is_zip_iterator_v = is_zip_iterator<Iter>::value;

}    // namespace hpx::traits
