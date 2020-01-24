//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_FUTURE_TUPLE_HPP
#define HPX_TRAITS_IS_FUTURE_TUPLE_HPP

#include <hpx/traits/is_future.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    template <typename Tuple, typename Enable = void>
    struct is_future_tuple
      : std::false_type
    {};

    template <typename ...Ts>
    struct is_future_tuple<util::tuple<Ts...> >
      : util::all_of<is_future<Ts>...>
    {};
}}

#endif
