//  Copyright Eric Niebler 2013-2015
//  Copyright 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tagged.hpp>
#include <hpx/datastructures/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename S>
    struct tagged_pair
      : tagged<std::pair<typename detail::tag_elem<F>::type,
                   typename detail::tag_elem<S>::type>,
            typename detail::tag_spec<F>::type,
            typename detail::tag_spec<S>::type>
    {
        typedef tagged<std::pair<typename detail::tag_elem<F>::type,
                           typename detail::tag_elem<S>::type>,
            typename detail::tag_spec<F>::type,
            typename detail::tag_spec<S>::type>
            base_type;

        template <typename... Ts>
        tagged_pair(Ts&&... ts)
          : base_type(std::forward<Ts>(ts)...)
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    constexpr HPX_FORCEINLINE tagged_pair<Tag1(typename std::decay<T1>::type),
        Tag2(typename std::decay<T2>::type)>
    make_tagged_pair(std::pair<T1, T2>&& p)
    {
        typedef tagged_pair<Tag1(typename std::decay<T1>::type),
            Tag2(typename std::decay<T2>::type)>
            result_type;

        return result_type(std::move(p));
    }

    template <typename Tag1, typename Tag2, typename T1, typename T2>
    constexpr HPX_FORCEINLINE tagged_pair<Tag1(typename std::decay<T1>::type),
        Tag2(typename std::decay<T2>::type)>
    make_tagged_pair(std::pair<T1, T2> const& p)
    {
        typedef tagged_pair<Tag1(typename std::decay<T1>::type),
            Tag2(typename std::decay<T2>::type)>
            result_type;

        return result_type(p);
    }

    template <typename Tag1, typename Tag2, typename... Ts>
    constexpr HPX_FORCEINLINE tagged_pair<
        Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
        Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>
    make_tagged_pair(hpx::tuple<Ts...>&& p)
    {
        static_assert(
            sizeof...(Ts) >= 2, "hpx::tuple must have at least 2 elements");

        typedef tagged_pair<
            Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
            Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>
            result_type;

        return result_type(std::move(get<0>(p)), std::move(get<1>(p)));
    }

    template <typename Tag1, typename Tag2, typename... Ts>
    constexpr HPX_FORCEINLINE tagged_pair<
        Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
        Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>
    make_tagged_pair(hpx::tuple<Ts...> const& p)
    {
        static_assert(
            sizeof...(Ts) >= 2, "hpx::tuple must have at least 2 elements");

        typedef tagged_pair<
            Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
            Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>
            result_type;

        return result_type(get<0>(p), get<1>(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    constexpr HPX_FORCEINLINE tagged_pair<Tag1(typename std::decay<T1>::type),
        Tag2(typename std::decay<T2>::type)>
    make_tagged_pair(T1&& t1, T2&& t2)
    {
        typedef tagged_pair<Tag1(typename std::decay<T1>::type),
            Tag2(typename std::decay<T2>::type)>
            result_type;

        return result_type(std::forward<T1>(t1), std::forward<T2>(t2));
    }
}}    // namespace hpx::util

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2>
    struct tuple_size<util::tagged_pair<Tag1, Tag2>>
      : std::integral_constant<std::size_t, 2>
    {
    };

    template <std::size_t N, typename Tag1, typename Tag2>
    struct tuple_element<N, util::tagged_pair<Tag1, Tag2>>
      : hpx::tuple_element<N,
            std::pair<typename util::detail::tag_elem<Tag1>::type,
                typename util::detail::tag_elem<Tag2>::type>>
    {
    };
}    // namespace hpx
