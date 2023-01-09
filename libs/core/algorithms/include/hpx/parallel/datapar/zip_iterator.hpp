//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/vector_pack_load_store.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/type_support/pack.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct is_data_aligned_impl<hpx::util::zip_iterator<Iter...>>
    {
        template <std::size_t... Is>
        static HPX_FORCEINLINE constexpr bool call(
            hpx::util::zip_iterator<Iter...> const& it,
            hpx::util::index_pack<Is...>)
        {
            auto const& t = it.get_iterator_tuple();
            return (true && ... && is_data_aligned(hpx::get<Is>(t)));
        }

        static HPX_FORCEINLINE constexpr bool call(
            hpx::util::zip_iterator<Iter...> const& it)
        {
            return call(it,
                typename hpx::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Iter>
    struct iterator_datapar_compatible_impl<hpx::util::zip_iterator<Iter...>>
      : hpx::util::all_of<std::is_arithmetic<
            typename std::iterator_traits<Iter>::value_type>...>
    {
    };
}    // namespace hpx::parallel::util::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::traits {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tuple, typename... Iter, std::size_t... Is>
        constexpr Tuple aligned_pack(
            hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::make_tuple(
                vector_pack_load<typename hpx::tuple_element<Is, Tuple>::type,
                    typename std::iterator_traits<Iter>::value_type>::
                    aligned(hpx::get<Is>(t))...);
        }

        template <typename Tuple, typename... Iter, std::size_t... Is>
        constexpr Tuple unaligned_pack(
            hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            return hpx::make_tuple(
                vector_pack_load<typename hpx::tuple_element<Is, Tuple>::type,
                    typename std::iterator_traits<Iter>::value_type>::
                    unaligned(hpx::get<Is>(t))...);
        }
    }    // namespace detail

    template <typename... Vector, typename ValueType>
    struct vector_pack_load<hpx::tuple<Vector...>, ValueType>
    {
        using value_type = hpx::tuple<Vector...>;

        template <typename... Iter>
        static constexpr value_type aligned(
            hpx::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::aligned_pack<value_type>(iter,
                typename hpx::util::make_index_pack<sizeof...(Iter)>::type());
        }

        template <typename... Iter>
        static constexpr value_type unaligned(
            hpx::util::zip_iterator<Iter...> const& iter)
        {
            return traits::detail::unaligned_pack<value_type>(iter,
                typename hpx::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Tuple, typename... Iter, std::size_t... Is>
        constexpr void aligned_pack(Tuple& value,
            hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            (vector_pack_store<typename hpx::tuple_element<Is, Tuple>::type,
                 typename std::iterator_traits<Iter>::value_type>::
                    aligned(hpx::get<Is>(value), hpx::get<Is>(t)),
                ...);
        }

        template <typename Tuple, typename... Iter, std::size_t... Is>
        constexpr void unaligned_pack(Tuple& value,
            hpx::util::zip_iterator<Iter...> const& iter,
            hpx::util::index_pack<Is...>)
        {
            auto const& t = iter.get_iterator_tuple();
            (vector_pack_store<typename hpx::tuple_element<Is, Tuple>::type,
                 typename std::iterator_traits<Iter>::value_type>::
                    unaligned(hpx::get<Is>(value), hpx::get<Is>(t)),
                ...);
        }
    }    // namespace detail

    template <typename... Vector, typename ValueType>
    struct vector_pack_store<hpx::tuple<Vector...>, ValueType>
    {
        template <typename V, typename... Iter>
        static constexpr void aligned(
            V& value, hpx::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::aligned_pack(value, iter,
                typename hpx::util::make_index_pack<sizeof...(Iter)>::type());
        }

        template <typename V, typename... Iter>
        static constexpr void unaligned(
            V& value, hpx::util::zip_iterator<Iter...> const& iter)
        {
            traits::detail::unaligned_pack(value, iter,
                typename hpx::util::make_index_pack<sizeof...(Iter)>::type());
        }
    };
}    // namespace hpx::parallel::traits

#endif
