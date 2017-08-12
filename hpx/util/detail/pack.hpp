//  Copyright (c) 2014-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_PACK_HPP
#define HPX_UTIL_DETAIL_PACK_HPP

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    struct empty {};

    template <typename ...Ts>
    struct pack
    {
        typedef pack type;
        static const std::size_t size = sizeof...(Ts);
    };

    template <typename T, T ...Vs>
    struct pack_c
    {
        typedef pack_c type;
        static const std::size_t size = sizeof...(Vs);
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Left, typename Right>
    struct make_index_pack_join;

    template <std::size_t... Left, std::size_t... Right>
    struct make_index_pack_join<
        pack_c<std::size_t, Left...>
      , pack_c<std::size_t, Right...>
    > : pack_c<std::size_t, Left..., (sizeof...(Left) + Right)...>
    {};

    template <std::size_t N>
    struct make_index_pack
      : make_index_pack_join<
            typename make_index_pack<N / 2>::type
          , typename make_index_pack<N - N / 2>::type
        >
    {};

    template <>
    struct make_index_pack<0>
      : pack_c<std::size_t>
    {};

    template <>
    struct make_index_pack<1>
      : pack_c<std::size_t, 0>
    {};

    ///////////////////////////////////////////////////////////////////////////
    // Workaround for clang bug [https://bugs.llvm.org/show_bug.cgi?id=35077]
    template <typename T>
    struct is_true
      : std::integral_constant<bool, (bool)T::value>
    {};

    template <typename T>
    struct is_false
      : std::integral_constant<bool, !(bool)T::value>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct _always_true
      : std::true_type
    {};

    template <typename ...Ts>
    struct _always_false
      : std::false_type
    {};

    template <typename ...Ts>
    struct all_of;

    template <bool ...Vs>
    struct all_of<pack_c<bool, Vs...> >
      : std::is_same<
            pack_c<bool, Vs...>
          , pack_c<bool, (Vs || true)...> // true...
        >
    {};

    template <typename ...Ts>
    static std::false_type _all_of(...);

    template <typename ...Ts>
    static auto _all_of(int) -> _always_true<
        typename std::enable_if<is_true<Ts>::value>::type...>;

    template <typename ...Ts>
    struct all_of
      : decltype(detail::_all_of<Ts...>(0))
    {};

    template <>
    struct all_of<> // <fake-type>
      : std::true_type
    {};

    template <typename ...Ts>
    struct any_of;

    template <bool ...Vs>
    struct any_of<pack_c<bool, Vs...> >
      : std::integral_constant<bool,
            !std::is_same<
                pack_c<bool, Vs...>
              , pack_c<bool, (Vs && false)...> // false...
            >::value
        >
    {};

    template <typename ...Ts>
    static std::true_type _any_of(...);

    template <typename ...Ts>
    static auto _any_of(int) -> _always_false<
        typename std::enable_if<is_false<Ts>::value>::type...>;

    template <typename ...Ts>
    struct any_of
      : decltype(detail::_any_of<Ts...>(0))
    {};

    template <>
    struct any_of<> // <fake-type>
      : std::false_type
    {};

    template <typename ...Ts>
    struct none_of
      : std::integral_constant<bool, !any_of<Ts...>::value>
    {};

    template <typename T, typename ...Ts>
    struct contains
      : any_of<std::is_same<T, Ts>...>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I, typename T>
    struct indexed
    {
        typedef T type;
    };

    template <typename Ts, typename Is>
    struct indexer;

    template <typename ...Ts, std::size_t ...Is>
    struct indexer<pack<Ts...>, pack_c<std::size_t, Is...>>
      : indexed<Is, Ts>...
    {};

    template <std::size_t I, typename Ts>
    struct at_index_impl
    {
        static empty check_(...);

        template <std::size_t J, typename T>
        static indexed<J, T> check_(indexed<J, T> const&);

        typedef decltype(check_<I>(indexer<
            Ts, typename make_index_pack<Ts::size>::type
        >())) type;
    };

    template <std::size_t I, typename ...Ts>
    struct at_index
      : at_index_impl<I, pack<Ts...> >::type
    {};
}}}

#endif
