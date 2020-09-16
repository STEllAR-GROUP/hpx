//  Copyright Eric Niebler 2013-2015
//  Copyright 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tagged.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/identity.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    struct tagged_tuple
      : tagged<tuple<typename detail::tag_elem<Ts>::type...>,
            typename detail::tag_spec<Ts>::type...>
    {
        typedef tagged<tuple<typename detail::tag_elem<Ts>::type...>,
            typename detail::tag_spec<Ts>::type...>
            base_type;

        template <typename... Ts_>
        tagged_tuple(Ts_&&... ts)
          : base_type(std::forward<Ts_>(ts)...)
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tag, typename T>
        struct tagged_type
        {
            typedef typename decay<T>::type decayed_type;
            typedef typename hpx::util::identity<Tag(decayed_type)>::type type;
        };
    }    // namespace detail

    template <typename... Tags, typename... Ts>
    constexpr HPX_FORCEINLINE
        tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
        make_tagged_tuple(Ts&&... ts)
    {
        typedef tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
            result_type;

        return result_type(std::forward<Ts>(ts)...);
    }

    template <typename... Tags, typename... Ts>
    constexpr HPX_FORCEINLINE
        tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
        make_tagged_tuple(tuple<Ts...>&& t)
    {
        static_assert(sizeof...(Tags) == tuple_size<tuple<Ts...>>::value,
            "the number of tags must be identical to the size of the given "
            "tuple");

        typedef tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
            result_type;

        return result_type(std::move(t));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Tag, std::size_t I, typename Tuple>
        struct tagged_element_type
        {
            typedef typename tuple_element<I, Tuple>::type element_type;
            typedef typename hpx::util::identity<Tag(element_type)>::type type;
        };

        template <typename Tuple, typename Indices, typename... Tags>
        struct tagged_tuple_helper;

        template <typename... Ts, std::size_t... Is, typename... Tags>
        struct tagged_tuple_helper<tuple<Ts...>, index_pack<Is...>, Tags...>
        {
            typedef tagged_tuple<
                typename tagged_element_type<Tags, Is, tuple<Ts...>>::type...>
                type;
        };
    }    // namespace detail
}}       // namespace hpx::util

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    struct tuple_size<util::tagged_tuple<Ts...>>
      : tuple_size<tuple<typename util::detail::tag_elem<Ts>::type...>>
    {
    };

    template <std::size_t N, typename... Ts>
    struct tuple_element<N, util::tagged_tuple<Ts...>>
      : tuple_element<N, tuple<typename util::detail::tag_elem<Ts>::type...>>
    {
    };
}    // namespace hpx
