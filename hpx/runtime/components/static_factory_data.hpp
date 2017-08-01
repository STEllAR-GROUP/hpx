//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Make HPX inspect tool happy: hpxinspect:nounnamed

#if !defined(HPX_COMPONENTS_STATIC_FACTORY_DATA_HPP)
#define HPX_COMPONENTS_STATIC_FACTORY_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/plugin/export_plugin.hpp>
#include <hpx/util/plugin/virtual_constructor.hpp>
#include <hpx/util/detail/pp/cat.hpp>

#include <map>
#include <string>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    struct static_factory_load_data_type
    {
        char const* name;     // component name
        hpx::util::plugin::get_plugins_list_type get_factory;
    };

    HPX_EXPORT void init_registry_module(static_factory_load_data_type const&);
    HPX_EXPORT void init_registry_factory(static_factory_load_data_type const&);
    HPX_EXPORT void init_registry_commandline(static_factory_load_data_type const&);
    HPX_EXPORT void init_registry_startup_shutdown(static_factory_load_data_type const&);
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DECLARE_FACTORY_STATIC(name, base)                                \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any>*       \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME(name, base)()                     \
/**/

#define HPX_DEFINE_FACTORY_STATIC(module, name, base)                         \
    {                                                                         \
        HPX_PP_STRINGIZE(module),                                             \
        HPX_PLUGIN_LIST_NAME(name, base)                                      \
    }                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_MODULE_STATIC(name, base)                           \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                   \
    namespace {                                                               \
        struct HPX_PP_CAT(init_registry_module_static_, name)                 \
        {                                                                     \
            HPX_PP_CAT(init_registry_module_static_, name)()                  \
            {                                                                 \
                hpx::components::static_factory_load_data_type data =         \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base);\
                hpx::components::init_registry_module(data);                  \
            }                                                                 \
        } HPX_PP_CAT(module_data_, __LINE__);                                 \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_FACTORY_STATIC(name, componentname, base)           \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                   \
    namespace {                                                               \
        struct HPX_PP_CAT(init_registry_factory_static_, componentname)       \
        {                                                                     \
            HPX_PP_CAT(init_registry_factory_static_, componentname)()        \
            {                                                                 \
                hpx::components::static_factory_load_data_type data =         \
                    HPX_DEFINE_FACTORY_STATIC(componentname, name, base);     \
                hpx::components::init_registry_factory(data);                 \
            }                                                                 \
        } HPX_PP_CAT(componentname, HPX_PP_CAT(_factory_data_, __LINE__));    \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_COMMANDLINE_STATIC(name, base)                      \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                   \
    namespace {                                                               \
        struct HPX_PP_CAT(init_registry_module_commandline_, name)            \
        {                                                                     \
            HPX_PP_CAT(init_registry_module_commandline_, name)()             \
            {                                                                 \
                hpx::components::static_factory_load_data_type data =         \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base);\
                hpx::components::init_registry_commandline(data);             \
            }                                                                 \
        } HPX_PP_CAT(module_commandline_data_, __LINE__);                     \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_INIT_REGISTRY_STARTUP_SHUTDOWN_STATIC(name, base)                 \
    HPX_DECLARE_FACTORY_STATIC(name, base);                                   \
    namespace {                                                               \
        struct HPX_PP_CAT(init_registry_module_startup_shutdown_, name)       \
        {                                                                     \
            HPX_PP_CAT(init_registry_module_startup_shutdown_, name)()        \
            {                                                                 \
                hpx::components::static_factory_load_data_type data =         \
                    HPX_DEFINE_FACTORY_STATIC(HPX_COMPONENT_NAME, name, base);\
                hpx::components::init_registry_startup_shutdown(data);        \
            }                                                                 \
        } HPX_PP_CAT(module_startup_shutdown_data_, __LINE__);                \
    }                                                                         \
    /**/

#endif

