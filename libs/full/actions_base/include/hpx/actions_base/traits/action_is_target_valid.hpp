//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for Action::component_type::is_target_valid
    namespace detail {

        struct is_target_valid_helper
        {
            // by default we return true if the given id is not referring to a
            // locality
            template <typename Action>
            static bool call(wrap_int, naming::id_type const& id) noexcept
            {
                // All component types requires valid id for its actions to be
                // invoked (by default)
                return !naming::is_locality(id);
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto call(int, naming::id_type const& id) noexcept
                -> decltype(Action::component_type::is_target_valid(id))
            {
                // by default we forward this to the component type
                using component_type = typename Action::component_type;
                return component_type::is_target_valid(id);
            }
        };
    }    // namespace detail

    template <typename Action, typename Enable = void>
    struct action_is_target_valid
    {
        static bool call(naming::id_type const& id) noexcept
        {
            return detail::is_target_valid_helper::template call<Action>(0, id);
        }
    };
}}    // namespace hpx::traits
