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
#include <hpx/datastructures/tagged.hpp>
#include <hpx/datastructures/tagged_tuple.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/type_support/decay.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace util {
    template <typename... Tags, typename... Ts>
    hpx::future<typename detail::tagged_tuple_helper<hpx::tuple<Ts...>,
        typename util::make_index_pack<sizeof...(Tags)>::type, Tags...>::type>
    make_tagged_tuple(hpx::future<hpx::tuple<Ts...>>&& f)
    {
        static_assert(
            sizeof...(Tags) == hpx::tuple_size<hpx::tuple<Ts...>>::value,
            "the number of tags must be identical to the size of the given "
            "tuple");

        typedef typename detail::tagged_tuple_helper<hpx::tuple<Ts...>,
            typename util::make_index_pack<sizeof...(Tags)>::type,
            Tags...>::type result_type;

        return lcos::make_future<result_type>(
            std::move(f), [](hpx::tuple<Ts...>&& t) -> result_type {
                return make_tagged_tuple<Tags...>(std::move(t));
            });
    }
}}    // namespace hpx::util
