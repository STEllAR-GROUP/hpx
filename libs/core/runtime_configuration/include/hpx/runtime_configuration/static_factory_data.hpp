//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Make HPX inspect tool happy: hpxinspect:nounnamed

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/modules/plugin.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <map>
#include <string>
#include <vector>

namespace hpx::components {

    ///////////////////////////////////////////////////////////////////////////
    struct static_factory_load_data_type
    {
        char const* name;    // component name
        hpx::util::plugin::get_plugins_list_type get_factory;
    };

    HPX_CORE_EXPORT bool& get_initial_static_loading() noexcept;

    HPX_CORE_EXPORT std::vector<static_factory_load_data_type>&
    get_static_module_data();
    HPX_CORE_EXPORT
    void init_registry_module(static_factory_load_data_type const&);

    HPX_CORE_EXPORT bool get_static_factory(
        std::string const& instance, util::plugin::get_plugins_list_type& f);
    HPX_CORE_EXPORT
    void init_registry_factory(static_factory_load_data_type const&);

    HPX_CORE_EXPORT bool get_static_commandline(
        std::string const& instance, util::plugin::get_plugins_list_type& f);
    HPX_CORE_EXPORT
    void init_registry_commandline(static_factory_load_data_type const&);

    HPX_CORE_EXPORT bool get_static_startup_shutdown(
        std::string const& instance, util::plugin::get_plugins_list_type& f);
    HPX_CORE_EXPORT
    void init_registry_startup_shutdown(static_factory_load_data_type const&);
}    // namespace hpx::components

////////////////////////////////////////////////////////////////////////////////
#define HPX_DECLARE_FACTORY_STATIC(name, base)                                 \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, hpx::any_nonser>*   \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME(name, base)() /**/

#define HPX_DEFINE_FACTORY_STATIC(module, name, base)                          \
    {                                                                          \
        HPX_PP_STRINGIZE(module), HPX_PLUGIN_LIST_NAME(name, base)             \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_MODULE_STATIC(name, base)                            \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_static_, name)                  \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_static_, name)()                   \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_module(data);                   \
            }                                                                  \
        } HPX_PP_CAT(module_data_, __LINE__);                                  \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_FACTORY_STATIC(name, componentname, base)            \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_factory_static_, componentname)        \
        {                                                                      \
            HPX_PP_CAT(init_registry_factory_static_, componentname)()         \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(componentname, name, base);      \
                hpx::components::init_registry_factory(data);                  \
            }                                                                  \
        } HPX_PP_CAT(componentname, HPX_PP_CAT(_factory_data_, __LINE__));     \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_COMMANDLINE_STATIC(name, base)                       \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_commandline_, name)             \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_commandline_, name)()              \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_commandline(data);              \
            }                                                                  \
        } HPX_PP_CAT(module_commandline_data_, __LINE__);                      \
    }                                                                          \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_STARTUP_SHUTDOWN_STATIC(name, base)                  \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                    \
    namespace {                                                                \
        struct HPX_PP_CAT(init_registry_module_startup_shutdown_, name)        \
        {                                                                      \
            HPX_PP_CAT(init_registry_module_startup_shutdown_, name)()         \
            {                                                                  \
                hpx::components::static_factory_load_data_type data =          \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base); \
                hpx::components::init_registry_startup_shutdown(data);         \
            }                                                                  \
        } HPX_PP_CAT(module_startup_shutdown_data_, __LINE__);                 \
    }                                                                          \
    /**/
