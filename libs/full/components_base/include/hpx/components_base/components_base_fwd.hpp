//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/traits/managed_component_policies.hpp>

namespace hpx {

    /// \namespace components
    namespace components {

        /// \ cond NODETAIL
        namespace detail {
            struct this_type
            {
            };
        }    // namespace detail
        /// \ endcond

        ///////////////////////////////////////////////////////////////////////
        class pinned_ptr;

        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        class fixed_component;

        template <typename Component>
        class component;

        template <typename Component, typename Derived = detail::this_type>
        class managed_component;

        ///////////////////////////////////////////////////////////////////////
        template <typename Component = detail::this_type>
        class component_base;

        template <typename Component = detail::this_type>
        class fixed_component_base;

        template <typename Component = detail::this_type>
        class abstract_component_base;

        template <typename Component>
        using abstract_simple_component_base HPX_DEPRECATED_V(1, 8,
            "The type hpx::components::abstract_simple_component_base is "
            "deprecated. Please use hpx::components::abstract_component_base "
            "instead.") = abstract_component_base<Component>;

        template <typename Component, typename Derived = detail::this_type>
        class abstract_managed_component_base;

        template <typename Component, typename Wrapper = detail::this_type,
            typename CtorPolicy = traits::construct_without_back_ptr,
            typename DtorPolicy = traits::managed_object_controls_lifetime>
        class managed_component_base;

    }    // namespace components
}    // namespace hpx
