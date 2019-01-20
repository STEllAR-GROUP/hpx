//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_BRACE_INITIALIZABLE_TRAITS_HPP
#define HPX_TRAITS_BRACE_INITIALIZABLE_TRAITS_HPP

#include <hpx/config/automatic_struct_serialization.hpp>
#include <hpx/util/unused.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace traits
{
#if defined(HPX_SUPPORT_AUTOMATIC_STRUCT_SERIALIZATION)
    struct wildcard
    {
        // Excluded hpx::util::unused_type from wildcard conversions
        // due to ambiguity (unused_type has own conversion to every type).
        template <typename T,
                typename =
                std::enable_if_t<!std::is_lvalue_reference<T>::value &&
                !std::is_same<typename std::decay<T>::type,
                              hpx::util::unused_type>::value>>
        operator T&&() const;

        template <typename T,
                typename =
                std::enable_if_t<std::is_copy_constructible<T>::value &&
                !std::is_same<typename std::decay<T>::type,
                              hpx::util::unused_type>::value>>
        operator T&() const;
    };

    template <std::size_t N = 0>
    static constexpr const wildcard _{};

    template <typename T, std::size_t ... I>
    inline constexpr auto
    is_brace_constructible_(std::index_sequence<I...>, T *)
        -> decltype(T{_<I>...}, std::true_type{})
    {
        return {};
    }

    template <std::size_t ... I>
    inline constexpr std::false_type
    is_brace_constructible_(std::index_sequence<I...>, ...)
    {
        return {};
    }


    template <typename T, std::size_t N>
    constexpr auto
    is_brace_constructible()
        -> decltype(is_brace_constructible_(std::make_index_sequence<N>{},
                                            static_cast<T *>(nullptr)))
    {
        return {};
    }

    template <typename T, typename U>
    struct is_paren_constructible_;

    template <typename T, std::size_t ... I>
    struct is_paren_constructible_<T, std::index_sequence<I...>>
        : std::is_constructible<T, decltype(_<I>)...>
    {
    };

    template <typename T, std::size_t N>
    constexpr auto
    is_paren_constructible()
        -> is_paren_constructible_<T, std::make_index_sequence<N>>
    {
        return {};
    }

    template <std::size_t N>
    using size = std::integral_constant<std::size_t, N>;

    template <typename T,
          typename = std::enable_if_t<
                       std::is_class<T>{} &&
                       std::is_empty<T>{}
                     >
          >
    constexpr size<0> arity()
    {
        return {};
    }

#define MAKE_ARITY_FUNC(count)                                          \
    template <typename T,                                               \
            typename = std::enable_if_t<                                \
                         is_brace_constructible<T, count>() &&          \
                         !is_brace_constructible<T, count+1>() &&       \
                         !is_paren_constructible<T, count>()            \
                       >                                                \
           >                                                            \
    constexpr size<count> arity()                                       \
    {                                                                   \
        return {};                                                      \
    }


    MAKE_ARITY_FUNC(1)
    MAKE_ARITY_FUNC(2)
    MAKE_ARITY_FUNC(3)
    MAKE_ARITY_FUNC(4)
    MAKE_ARITY_FUNC(5)
    MAKE_ARITY_FUNC(6)
    MAKE_ARITY_FUNC(7)
    MAKE_ARITY_FUNC(8)
    MAKE_ARITY_FUNC(9)
    MAKE_ARITY_FUNC(10)
    MAKE_ARITY_FUNC(11)
    MAKE_ARITY_FUNC(12)
    MAKE_ARITY_FUNC(13)
    MAKE_ARITY_FUNC(14)
    MAKE_ARITY_FUNC(15)

#endif
}}

#endif
