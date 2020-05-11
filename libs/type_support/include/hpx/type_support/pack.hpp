//  Copyright (c) 2014-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace util {

    template <typename... Ts>
    struct pack
    {
        typedef pack type;
        static const std::size_t size = sizeof...(Ts);
    };

    template <typename T, T... Vs>
    struct pack_c
    {
        typedef pack_c type;
        static const std::size_t size = sizeof...(Vs);
    };

    template <std::size_t... Is>
    using index_pack = pack_c<std::size_t, Is...>;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Left, typename Right>
        struct make_index_pack_join;

        template <std::size_t... Left, std::size_t... Right>
        struct make_index_pack_join<index_pack<Left...>, index_pack<Right...>>
          : index_pack<Left..., (sizeof...(Left) + Right)...>
        {
        };
    }    // namespace detail

    template <std::size_t N>
    struct make_index_pack
#if defined(HPX_HAVE_BUILTIN_INTEGER_PACK)
      : index_pack<__integer_pack(N)...>
#elif defined(HPX_HAVE_BUILTIN_MAKE_INTEGER_SEQ)
      : __make_integer_seq<pack_c, std::size_t, N>
#else
      : detail::make_index_pack_join<typename make_index_pack<N / 2>::type,
            typename make_index_pack<N - N / 2>::type>
#endif
    {
    };

    template <>
    struct make_index_pack<0> : pack_c<std::size_t>
    {
    };

    template <>
    struct make_index_pack<1> : index_pack<0>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // Workaround for clang bug [https://bugs.llvm.org/show_bug.cgi?id=35077]
    namespace detail {
        template <typename T>
        struct is_true : std::integral_constant<bool, (bool) T::value>
        {
        };

        template <typename T>
        struct is_false : std::integral_constant<bool, !(bool) T::value>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename... Ts>
        struct always_true : std::true_type
        {
        };

        template <typename... Ts>
        struct always_false : std::false_type
        {
        };

        template <typename... Ts>
        static std::false_type all_of(...);

        template <typename... Ts>
        static auto all_of(int) -> always_true<
            typename std::enable_if<is_true<Ts>::value>::type...>;
    }    // namespace detail

    template <typename... Ts>
    struct all_of : decltype(detail::all_of<Ts...>(0))
    {
    };

    template <>
    struct all_of<>    // <fake-type>
      : std::true_type
    {
    };

    namespace detail {
        template <typename... Ts>
        static std::true_type any_of(...);

        template <typename... Ts>
        static auto any_of(int) -> always_false<
            typename std::enable_if<is_false<Ts>::value>::type...>;
    }    // namespace detail

    template <typename... Ts>
    struct any_of : decltype(detail::any_of<Ts...>(0))
    {
    };

    template <>
    struct any_of<>    // <fake-type>
      : std::false_type
    {
    };

    template <typename... Ts>
    struct none_of : std::integral_constant<bool, !any_of<Ts...>::value>
    {
    };

    template <typename T, typename... Ts>
    struct contains : any_of<std::is_same<T, Ts>...>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        struct empty
        {
        };

#if defined(HPX_HAVE_BUILTIN_TYPE_PACK_ELEMENT)
        template <std::size_t I, typename Ts, bool C = (I < Ts::size)>
        struct at_index_impl
        {
            using type = empty;
        };

        template <std::size_t I, typename... Ts>
        struct at_index_impl<I, pack<Ts...>, true>
        {
            using type = struct unspecified
            {
                using type = __type_pack_element<I, Ts...>;
            };
        };
#else
        template <std::size_t I, typename T>
        struct indexed
        {
            typedef T type;
        };

        template <typename Ts, typename Is>
        struct indexer;

        template <typename... Ts, std::size_t... Is>
        struct indexer<pack<Ts...>, pack_c<std::size_t, Is...>>
          : indexed<Is, Ts>...
        {
        };

        template <std::size_t I, typename Ts>
        struct at_index_impl
        {
            static empty check_(...);

            template <std::size_t J, typename T>
            static indexed<J, T> check_(indexed<J, T> const&);

            typedef decltype(check_<I>(
                indexer<Ts, typename make_index_pack<Ts::size>::type>())) type;
        };
#endif
    }    // namespace detail

    template <std::size_t I, typename... Ts>
    struct at_index : detail::at_index_impl<I, pack<Ts...>>::type
    {
    };

}}    // namespace hpx::util
