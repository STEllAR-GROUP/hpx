//  Copyright (c) 2014-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_PACK_HPP
#define HPX_UTIL_DETAIL_PACK_HPP

#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>

#include <cstddef>

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
    template <typename ...Ts>
    struct all_of;

    template <bool ...Vs>
    struct all_of<pack_c<bool, Vs...> >
      : boost::mpl::bool_<
            boost::is_same<
                pack_c<bool, Vs...>
              , pack_c<bool, (Vs || true)...> // true...
            >::value
        >
    {};

    template <typename ...Ts>
    struct all_of
      : all_of<pack_c<bool, ((bool)Ts::value)...> >
    {};

    template <>
    struct all_of<> // <fake-type>
      : boost::mpl::true_
    {};

    template <typename ...Ts>
    struct any_of;

    template <bool ...Vs>
    struct any_of<pack_c<bool, Vs...> >
      : boost::mpl::bool_<
            !all_of<pack_c<bool, !Vs...> >::value
        >
    {};

    template <typename ...Ts>
    struct any_of
      : any_of<pack_c<bool, ((bool)Ts::value)...> >
    {};

    template <>
    struct any_of<> // <fake-type>
      : boost::mpl::false_
    {};

    template <typename ...Ts>
    struct none_of;

    template <bool ...Vs>
    struct none_of<pack_c<bool, Vs...> >
      : all_of<pack_c<bool, !Vs...> >
    {};

    template <typename ...Ts>
    struct none_of
      : none_of<pack_c<bool, ((bool)Ts::value)...> >
    {};

    template <>
    struct none_of<> // <fake-type>
      : boost::mpl::true_
    {};

    template <typename T, typename ...Ts>
    struct contains
      : any_of<boost::is_same<T, Ts>...>
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
