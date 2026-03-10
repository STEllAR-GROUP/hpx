//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file components_base_fwd.hpp
/// \page hpx::components::component, hpx::components::component_base
/// \headerfile hpx/components.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX_MODULES) && !defined(HPX_FULL_EXPORTS) &&             \
    !defined(HPX_COMPILE_BMI)
import HPX.Full;
#else

#include <hpx/components_base/traits/managed_component_policies.hpp>

namespace hpx::components {

    ///////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT class pinned_ptr;

    ///////////////////////////////////////////////////////////////////////
    template <typename Component>
    HPX_CXX_EXPORT class fixed_component;

    /// The \a component class wraps around a given component type, adding
    /// additional type aliases and constructors. It inherits from the
    /// specified component type.
    template <typename Component>
    HPX_CXX_EXPORT class component;

    template <typename Component, typename Derived = void>
    HPX_CXX_EXPORT class managed_component;

    ///////////////////////////////////////////////////////////////////////
    /// \a component_base serves as a base class for components. It provides
    /// common functionality needed by components, such as address and ID
    /// retrieval. The template parameter \a Component specifies the
    /// derived component type.
    template <typename Component = void>
    HPX_CXX_EXPORT class component_base;

    template <typename Component = void>
    HPX_CXX_EXPORT class fixed_component_base;

    template <typename Component = void>
    HPX_CXX_EXPORT class abstract_component_base;

    template <typename Component, typename Derived = void>
    HPX_CXX_EXPORT class abstract_managed_component_base;

    template <typename Component, typename Wrapper = void,
        typename CtorPolicy = traits::construct_without_back_ptr,
        typename DtorPolicy = traits::managed_object_controls_lifetime>
    HPX_CXX_EXPORT class managed_component_base;
}    // namespace hpx::components

#endif
