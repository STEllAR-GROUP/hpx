// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2012 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXPORT_PLUGIN_VP_2004_08_25
#define HPX_EXPORT_PLUGIN_VP_2004_08_25

#include <string>
#include <map>

#include <boost/config.hpp>
#include <boost/any.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <hpx/util/plugin/config.hpp>
#include <hpx/util/plugin/concrete_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_NAME_2(name1, name2)                                       \
    BOOST_PP_CAT(name1, BOOST_PP_CAT(_, name2))                               \
    /**/

#define HPX_PLUGIN_NAME_3(name, base, cname)                                  \
    BOOST_PP_CAT(cname, BOOST_PP_CAT(_,                                       \
        BOOST_PP_CAT(name, BOOST_PP_CAT(_, base))))                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_LIST_NAME(name, base)                                      \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(HPX_PLUGIN_SYMBOLS_PREFIX, _exported_plugins_list_),     \
        HPX_PLUGIN_NAME_2(name, base))                                        \
    /**/

#define HPX_PLUGIN_EXPORTER_NAME(name, base, cname)                           \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(HPX_PLUGIN_SYMBOLS_PREFIX, _plugin_exporter_),           \
        HPX_PLUGIN_NAME_3(name, base, cname))                                 \
    /**/

#define HPX_PLUGIN_EXPORTER_INSTANCE_NAME(name, base, cname)                  \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(HPX_PLUGIN_SYMBOLS_PREFIX, _plugin_exporter_instance_),  \
        HPX_PLUGIN_NAME_3(name, base, cname))                                 \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_FORCE_LOAD_NAME(name, base)                                \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(HPX_PLUGIN_SYMBOLS_PREFIX, _exported_plugins_force_load_), \
        HPX_PLUGIN_NAME_2(name, base))                                        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_EXPORT(name, BaseType, ActualType, actualname, classname)  \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any> *      \
               HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME(name, classname)();        \
                                                                              \
    namespace {                                                               \
        struct HPX_PLUGIN_EXPORTER_NAME(name, actualname, classname) {        \
            HPX_PLUGIN_EXPORTER_NAME(name, actualname, classname)()           \
            {                                                                 \
                static hpx::util::plugin::concrete_factory<BaseType, ActualType > cf; \
                hpx::util::plugin::abstract_factory<BaseType>* w = &cf;       \
                std::string actname(BOOST_PP_STRINGIZE(actualname));          \
                boost::algorithm::to_lower(actname);                          \
                HPX_PLUGIN_LIST_NAME(name, classname)()->insert(              \
                    std::make_pair(actname, w));                              \
            }                                                                 \
        } HPX_PLUGIN_EXPORTER_INSTANCE_NAME(name, actualname, classname);     \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_EXPORT_LIST(name, classname)                               \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any> *      \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME(name, classname)()                \
    {                                                                         \
        static std::map<std::string, boost::any> r;                           \
        return &r;                                                            \
    }                                                                         \
    extern "C" HPX_PLUGIN_EXPORT_API                                          \
        void HPX_PLUGIN_FORCE_LOAD_NAME(name, classname)()                    \
    {                                                                         \
        HPX_PLUGIN_LIST_NAME(name, classname)();                              \
    }                                                                         \
    /**/

#endif

