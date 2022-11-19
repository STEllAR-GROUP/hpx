//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_colocated/functional/colocated_helpers.hpp>
#include <hpx/async_distributed/bind_action.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/pack.hpp>

namespace hpx { namespace detail {

    template <typename Action, typename Ts = typename Action::arguments_type>
    struct post_colocated_bound_action;

    template <typename Action, typename... Ts>
    struct post_colocated_bound_action<Action, hpx::tuple<Ts...>>
    {
        using type = hpx::detail::bound_action<Action,
            hpx::util::make_index_pack<1 + sizeof...(Ts)>,
            hpx::detail::bound<hpx::util::functional::extract_locality,
                hpx::util::index_pack<0, 1>, hpx::detail::placeholder<2ul>,
                hpx::id_type>,
            Ts...>;
    };
}}    // namespace hpx::detail

#define HPX_REGISTER_APPLY_COLOCATED_DECLARATION(Action, Name)
#define HPX_REGISTER_APPLY_COLOCATED(action, name)
