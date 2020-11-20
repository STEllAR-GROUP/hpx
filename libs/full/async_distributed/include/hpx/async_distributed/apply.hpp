//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_local/apply.hpp>
#include <hpx/util/bind_action.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail {
    // bound action
    template <typename Bound>
    struct apply_dispatch<Bound,
        typename std::enable_if<traits::is_bound_action<Bound>::value>::type>
    {
        template <typename Action, typename Is, typename... Ts, typename... Us>
        HPX_FORCEINLINE static bool call(
            hpx::util::detail::bound_action<Action, Is, Ts...> const& bound,
            Us&&... vs)
        {
            return bound.apply(std::forward<Us>(vs)...);
        }
    };
}}    // namespace hpx::detail
