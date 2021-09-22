//  Copyright Eric Niebler 2013-2015
//  Copyright 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tagged_pair.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    hpx::future<tagged_pair<Tag1(typename std::decay<T1>::type),
        Tag2(typename std::decay<T2>::type)>>
    make_tagged_pair(hpx::future<std::pair<T1, T2>>&& f)
    {
        typedef hpx::util::tagged_pair<Tag1(typename std::decay<T1>::type),
            Tag2(typename std::decay<T2>::type)>
            result_type;

#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(f);
        HPX_ASSERT(false);
        return hpx::future<result_type>();
#else
        return lcos::make_future<result_type>(
            std::move(f), [](std::pair<T1, T2>&& p) -> result_type {
                return make_tagged_pair<Tag1, Tag2>(std::move(p));
            });
#endif
    }

    template <typename Tag1, typename Tag2, typename... Ts>
    hpx::future<tagged_pair<
        Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
        Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>>
    make_tagged_pair(hpx::future<hpx::tuple<Ts...>>&& f)
    {
        static_assert(
            sizeof...(Ts) >= 2, "tuple must have at least 2 elements");

        typedef hpx::util::tagged_pair<
            Tag1(typename hpx::tuple_element<0, hpx::tuple<Ts...>>::type),
            Tag2(typename hpx::tuple_element<1, hpx::tuple<Ts...>>::type)>
            result_type;

#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(f);
        HPX_ASSERT(false);
        return hpx::future<result_type>();
#else
        return lcos::make_future<result_type>(
            std::move(f), [](hpx::tuple<Ts...>&& p) -> result_type {
                return make_tagged_pair<Tag1, Tag2>(std::move(p));
            });
#endif
    }
}}    // namespace hpx::util
