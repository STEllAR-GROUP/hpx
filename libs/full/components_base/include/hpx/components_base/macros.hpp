//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2026 Arpit Khandelwal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/preprocessor.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_GET_COMPONENT_TYPE(component)                               \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        HPX_ALWAYS_EXPORT components::component_type                           \
        component_type_database<component>::get() noexcept                     \
        {                                                                      \
            return value;                                                      \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT void component_type_database<component>::set(        \
            components::component_type t)                                      \
        {                                                                      \
            value = t;                                                         \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_TEMPLATE(template_, component)           \
    namespace hpx::traits {                                                    \
        HPX_PP_STRIP_PARENS(template_)                                         \
        struct component_type_database<HPX_PP_STRIP_PARENS(component)>         \
        {                                                                      \
            static components::component_type value;                           \
                                                                               \
            HPX_ALWAYS_EXPORT static components::component_type get() noexcept \
            {                                                                  \
                return value;                                                  \
            }                                                                  \
            HPX_ALWAYS_EXPORT static void set(components::component_type t)    \
            {                                                                  \
                value = t;                                                     \
            }                                                                  \
        };                                                                     \
                                                                               \
        HPX_PP_STRIP_PARENS(template_)                                         \
        components::component_type                                             \
            component_type_database<HPX_PP_STRIP_PARENS(component)>::value =   \
                to_int(hpx::components::component_enum_type::invalid);         \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_COMPONENT_TYPE_STATIC(component, type)                  \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        HPX_ALWAYS_EXPORT components::component_type                           \
        component_type_database<component>::get() noexcept                     \
        {                                                                      \
            return type;                                                       \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT void component_type_database<component>::set(        \
            components::component_type)                                        \
        {                                                                      \
            HPX_ASSERT(false);                                                 \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME(...) HPX_DEFINE_COMPONENT_NAME_(__VA_ARGS__)

#define HPX_DEFINE_COMPONENT_NAME_(...)                                        \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                  \
        HPX_DEFINE_COMPONENT_NAME_, HPX_PP_NARGS(__VA_ARGS__))(__VA_ARGS__))   \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_2(Component, name)                           \
    namespace hpx::components {                                                \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_name<Component, void>() noexcept                         \
        {                                                                      \
            return HPX_PP_STRINGIZE(name);                                     \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_base_name<Component, void>() noexcept                    \
        {                                                                      \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_NAME_3(Component, name, base_name)                \
    namespace hpx::components {                                                \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_name<Component, void>() noexcept                         \
        {                                                                      \
            return HPX_PP_STRINGIZE(name);                                     \
        }                                                                      \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_component_base_name<Component, void>() noexcept                    \
        {                                                                      \
            return base_name;                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#define HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)         \
    namespace hpx::components::commandline_options_provider {                  \
        hpx::program_options::options_description add_commandline_options()    \
        {                                                                      \
            return add_options_function();                                     \
        }                                                                      \
    }                                                                          \
    /***/

#define HPX_REGISTER_COMMANDLINE_MODULE(add_options_function)                  \
    HPX_REGISTER_COMMANDLINE_OPTIONS()                                         \
    HPX_REGISTER_COMMANDLINE_REGISTRY(                                         \
        hpx::components::component_commandline, commandline_options)           \
    HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)             \
    /**/

#define HPX_REGISTER_COMMANDLINE_MODULE_DYNAMIC(add_options_function)          \
    HPX_REGISTER_COMMANDLINE_OPTIONS_DYNAMIC()                                 \
    HPX_REGISTER_COMMANDLINE_REGISTRY_DYNAMIC(                                 \
        hpx::components::component_commandline, commandline_options)           \
    HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)             \
    /**/

#define HPX_REGISTER_COMPONENT_HEAP(Component)                                 \
    namespace hpx::components::detail {                                        \
        template <>                                                            \
        HPX_ALWAYS_EXPORT Component::heap_type&                                \
        component_heap_helper<Component>(...)                                  \
        {                                                                      \
            util::reinitializable_static<Component::heap_type> heap;           \
            return heap.get();                                                 \
        }                                                                      \
    }                                                                          \
    /**/
