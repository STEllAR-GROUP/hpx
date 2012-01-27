// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2012 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_EXPORT_PLUGIN_VP_2004_08_25
#define BOOST_EXPORT_PLUGIN_VP_2004_08_25

#include <string>
#include <map>

#include <boost/config.hpp>
#include <boost/any.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <boost/plugin/config.hpp>
#include <boost/plugin/concrete_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PLUGIN_NAME_2(name1, name2)                                     \
    BOOST_PP_CAT(name1, BOOST_PP_CAT(_, name2))                               \
    /**/

#define BOOST_PLUGIN_NAME_3(name, base, cname)                                \
    BOOST_PP_CAT(cname, BOOST_PP_CAT(_,                                       \
        BOOST_PP_CAT(name, BOOST_PP_CAT(_, base))))                           \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PLUGIN_LIST_NAME(name, base)                                    \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(BOOST_PLUGIN_PREFIX, _exported_plugins_list_),           \
        BOOST_PLUGIN_NAME_2(name, base))                                      \
    /**/

#define BOOST_PLUGIN_EXPORTER_NAME(name, base, cname)                         \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(BOOST_PLUGIN_PREFIX, _plugin_exporter_),                 \
        BOOST_PLUGIN_NAME_3(name, base, cname))                               \
    /**/

#define BOOST_PLUGIN_EXPORTER_INSTANCE_NAME(name, base, cname)                \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(BOOST_PLUGIN_PREFIX, _plugin_exporter_instance_),        \
        BOOST_PLUGIN_NAME_3(name, base, cname))                               \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PLUGIN_FORCE_LOAD_NAME(name, base)                              \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(BOOST_PLUGIN_PREFIX, _exported_plugins_force_load_),     \
        BOOST_PLUGIN_NAME_2(name, base))                                      \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PLUGIN_EXPORT(name, BaseType, ActualType, actualname, classname)\
    extern "C" BOOST_PLUGIN_EXPORT_API std::map<std::string, boost::any>&     \
               BOOST_PLUGIN_API BOOST_PLUGIN_LIST_NAME(name, classname)();    \
                                                                              \
    namespace {                                                               \
        struct BOOST_PLUGIN_EXPORTER_NAME(name, actualname, classname) {      \
            BOOST_PLUGIN_EXPORTER_NAME(name, actualname, classname)()         \
            {                                                                 \
                static boost::plugin::concrete_factory<BaseType, ActualType > cf; \
                boost::plugin::abstract_factory<BaseType>* w = &cf;           \
                std::string actname(BOOST_PP_STRINGIZE(actualname));          \
                boost::algorithm::to_lower(actname);                          \
                BOOST_PLUGIN_LIST_NAME(name, classname)().insert(             \
                    std::make_pair(actname, w));                              \
            }                                                                 \
        } BOOST_PLUGIN_EXPORTER_INSTANCE_NAME(name, actualname, classname);   \
    }                                                                         \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define BOOST_PLUGIN_EXPORT_LIST(name, classname)                             \
    extern "C" BOOST_PLUGIN_EXPORT_API std::map<std::string, boost::any>&     \
        BOOST_PLUGIN_API BOOST_PLUGIN_LIST_NAME(name, classname)()            \
    {                                                                         \
        static std::map<std::string, boost::any> r;                           \
        return r;                                                             \
    }                                                                         \
    extern "C" BOOST_PLUGIN_EXPORT_API                                        \
        void BOOST_PLUGIN_FORCE_LOAD_NAME(name, classname)()                  \
    {                                                                         \
        BOOST_PLUGIN_LIST_NAME(name, classname)();                            \
    }                                                                         \
    /**/

#endif

