//  Copyright (c) 2005-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STATIC_COMPONENT_DATA_HPP)
#define HPX_STATIC_COMPONENT_DATA_HPP

#include <hpx/config.hpp>
#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

#include <boost/preprocessor/stringize.hpp>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    struct static_component_load_data_type
    {
        char const* name;               // component name
        void (*force_load)();           // function to force linking component
        hpx::util::plugin::get_plugins_list_type get_factory;
    };

    struct static_module_load_data_type
    {
        char const* name;               // module name
        void (*force_load)();           // function to force linking module
    };
}}

#if HPX_STATIC_LINKING
///////////////////////////////////////////////////////////////////////////////
#define HPX_STATIC_DECLARE_COMPONENT(prefix, name)                            \
    extern "C" void HPX_PLUGIN_FORCE_LOAD_NAME(prefix, name)();               \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any>*       \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME(prefix, name)()                   \
/**/

#define HPX_STATIC_DEFINE_COMPONENT(prefix, name)                             \
    {                                                                         \
        BOOST_PP_STRINGIZE(name),                                             \
        HPX_PLUGIN_FORCE_LOAD_NAME(prefix, name),                             \
        HPX_PLUGIN_LIST_NAME(prefix, name)                                    \
    }                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_STATIC_DECLARE_MODULE(prefix, name)                               \
    extern "C" void HPX_PLUGIN_FORCE_LOAD_NAME(prefix, name)();               \
/**/

#define HPX_STATIC_DEFINE_MODULE(prefix, name)                                \
    {                                                                         \
        BOOST_PP_STRINGIZE(name),                                             \
        HPX_PLUGIN_FORCE_LOAD_NAME(prefix, name)                              \
    }                                                                         \
/**/

///////////////////////////////////////////////////////////////////////////////
// prototypes of functions used to force linking of components
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, binpacking_factory);
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, dataflow);
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, dataflow_trigger);
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, distributing_factory);
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, output_stream_factory);
HPX_STATIC_DECLARE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, remote_object);

// prototypes of functions used to force linking of modules
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, binpacking_factory);
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, dataflow);
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, distributing_factory);
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, iostreams);
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, memory);
HPX_STATIC_DECLARE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, remote_object);

#endif

///////////////////////////////////////////////////////////////////////////////
// table of components to link statically
namespace hpx { namespace components
{
    static static_component_load_data_type const static_component_load_data[] =
    {
    #if defined(HPX_STATIC_LINKING)
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, binpacking_factory),
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, dataflow),
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, dataflow_trigger),
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, distributing_factory),
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, output_stream_factory),
        HPX_STATIC_DEFINE_COMPONENT(HPX_PLUGIN_COMPONENT_PREFIX, remote_object),
    #endif
        { NULL, NULL }
    };

    // table of modules to link statically
    static static_module_load_data_type const static_module_load_data[] =
    {
    #if defined(HPX_STATIC_LINKING)
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, binpacking_factory),
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, dataflow),
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, distributing_factory),
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, iostreams),
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, memory),
        HPX_STATIC_DEFINE_MODULE(HPX_PLUGIN_COMPONENT_PREFIX, remote_object),
    #endif
        { NULL, NULL }
    };
}}

///////////////////////////////////////////////////////////////////////////////
// enable auto-linking on supported platforms
#if defined(BOOST_MSVC) && defined(HPX_STATIC_LINKING)
// auto-link modules
    #define HPX_AUTOLINK_LIB_NAME "binpacking_factory"
    #include <hpx/config/autolink.hpp>
    #define HPX_AUTOLINK_LIB_NAME "dataflow"
    #include <hpx/config/autolink.hpp>
    #define HPX_AUTOLINK_LIB_NAME "distributing_factory"
    #include <hpx/config/autolink.hpp>
    #define HPX_AUTOLINK_LIB_NAME "iostreams"
    #include <hpx/config/autolink.hpp>
    #define HPX_AUTOLINK_LIB_NAME "memory"
    #include <hpx/config/autolink.hpp>
    #define HPX_AUTOLINK_LIB_NAME "remote_object"
    #include <hpx/config/autolink.hpp>
#endif

#endif

