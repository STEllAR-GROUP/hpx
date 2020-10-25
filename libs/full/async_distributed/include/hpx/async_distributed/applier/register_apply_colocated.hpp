//  Copyright (c) 2014 thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>

namespace hpx { namespace detail {
    template <typename Action, typename Ts = typename Action::arguments_type>
    struct apply_colocated_bound_action;

    template <typename Action, typename... Ts>
    struct apply_colocated_bound_action<Action, hpx::tuple<Ts...>>
    {
        typedef hpx::util::detail::bound_action<Action,
            hpx::util::make_index_pack<1 + sizeof...(Ts)>,
            hpx::util::detail::bound<hpx::util::functional::extract_locality,
                hpx::util::index_pack<0, 1>,
                hpx::util::detail::placeholder<2ul>, hpx::id_type>,
            Ts...>
            type;
    };
}}    // namespace hpx::detail

#define HPX_REGISTER_APPLY_COLOCATED_DECLARATION(Action, Name)

/*
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(                            \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            typename hpx::detail::apply_colocated_bound_action<Action>::type  \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
    */
/**/

#define HPX_REGISTER_APPLY_COLOCATED(action, name)

/*
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION(                                        \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            typename hpx::detail::apply_colocated_bound_action<Action>::type  \
        >)                                                                    \
      , name                                                                  \
    );                                                                        \
    */
/**/
