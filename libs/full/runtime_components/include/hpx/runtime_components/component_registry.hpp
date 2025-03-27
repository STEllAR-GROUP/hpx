//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2017      Thomas Heller
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_colocated/server/destroy_component.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/traits/component_config_data.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/expand.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime_configuration/component_registry_base.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::components {

    namespace detail {

        HPX_EXPORT void get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath, bool is_static, char const* name,
            char const* component_string, factory_state state,
            char const* more);

        HPX_EXPORT bool is_component_enabled(char const* name);
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_registry provides a minimal implementation of a
    /// component's registry. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required
    /// functions to be exposed by a component's registry instance.
    ///
    /// \tparam Component   The component type this registry should be
    ///                     responsible for.
    template <typename Component, factory_state state>
    struct component_registry : component_registry_base
    {
        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all components implemented in this
        ///                 module.
        /// \param filepath
        /// \param is_static
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        bool get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath, bool is_static = false) override
        {
            using type_holder = typename Component::type_holder;

            char const* name = get_component_name<type_holder>();
            char const* more = traits::component_config_data<Component>::call();
            detail::get_component_info(fillini, filepath, is_static, name,
                HPX_COMPONENT_STRING, state, more);
            return true;
        }

        /// \brief Enables this type of registry and sets its destroy mechanism
        void register_component_type() override
        {
            using type_holder = typename Component::type_holder;
            using base_type_holder = typename Component::base_type_holder;

            char const* name = get_component_name<type_holder>();
            bool const enabled = detail::is_component_enabled(name);

            component_type type = components::get_component_type<type_holder>();
            component_type const base_type =
                components::get_component_type<base_type_holder>();
            if (to_int(hpx::components::component_enum_type::invalid) == type)
            {
                // First call to get_component_type, ask AGAS for a unique id.
                type = detail::get_agas_component_type(name,
                    components::get_component_base_name<type_holder>(),
                    base_type, enabled);
                components::set_component_type<type_holder>(type);
            }
            components::enabled(type) = enabled;
            components::deleter(type) = &server::destroy<Component>;
        }
    };
}    // namespace hpx::components

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component registry with
/// Hpx.Plugin.

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY(...)                           \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(__VA_ARGS__)                      \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(...)                          \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_,         \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_2(                             \
    ComponentType, componentname)                                              \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                 \
        ComponentType, componentname, ::hpx::components::factory_state::check) \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                             \
    ComponentType, componentname, state)                                       \
    using componentname##_component_registry_type =                            \
        hpx::components::component_registry<ComponentType, state>;             \
    HPX_REGISTER_COMPONENT_REGISTRY(                                           \
        componentname##_component_registry_type, componentname)                \
    template struct hpx::components::component_registry<ComponentType, state>; \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC(...)                   \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_(__VA_ARGS__)              \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_(...)                  \
    HPX_PP_EXPAND(HPX_PP_CAT(HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_, \
        HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))                               \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_2(                     \
    ComponentType, componentname)                                              \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                         \
        ComponentType, componentname, ::hpx::components::factory_state::check) \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                     \
    ComponentType, componentname, state)                                       \
    using componentname##_component_registry_type =                            \
        hpx::components::component_registry<ComponentType, state>;             \
    HPX_REGISTER_COMPONENT_REGISTRY_DYNAMIC(                                   \
        componentname##_component_registry_type, componentname)                \
    template struct hpx::components::component_registry<ComponentType, state>; \
/**/
