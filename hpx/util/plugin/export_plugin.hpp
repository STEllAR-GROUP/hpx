// Copyright Vladimir Prus 2004.
// Copyright (c) 2005-2014 Hartmut Kaiser
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Make HPX inspect tool happy: hpxinspect:nounnamed

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
#define HPX_PLUGIN_LIST_NAME_(prefix, name, base)                             \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(prefix, _exported_plugins_list_),                        \
            HPX_PLUGIN_NAME_2(name, base))                                    \
    /**/

#define HPX_PLUGIN_EXPORTER_NAME_(prefix, name, base, cname)                  \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(prefix, _plugin_exporter_),                              \
            HPX_PLUGIN_NAME_3(name, base, cname))                             \
    /**/

#define HPX_PLUGIN_EXPORTER_INSTANCE_NAME_(prefix, name, base, cname)         \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(prefix, _plugin_exporter_instance_),                     \
            HPX_PLUGIN_NAME_3(name, base, cname))                             \
    /**/

#define HPX_PLUGIN_FORCE_LOAD_NAME_(prefix, name, base)                       \
    BOOST_PP_CAT(                                                             \
        BOOST_PP_CAT(prefix, _exported_plugins_force_load_),                  \
            HPX_PLUGIN_NAME_2(name, base))                                    \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_LIST_NAME(name, base)                                      \
    HPX_PLUGIN_LIST_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX, name, base)              \
    /**/

#define HPX_PLUGIN_EXPORTER_NAME(name, base, cname)                           \
    HPX_PLUGIN_EXPORTER_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX, name, base, cname)   \
    /**/

#define HPX_PLUGIN_EXPORTER_INSTANCE_NAME(name, base, cname)                  \
    HPX_PLUGIN_EXPORTER_INSTANCE_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX,             \
        name, base, cname)                                                    \
    /**/

#define HPX_PLUGIN_FORCE_LOAD_NAME(name, base)                                \
    HPX_PLUGIN_FORCE_LOAD_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX, name, base)        \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_LIST_NAME_DYNAMIC(name, base)                              \
    HPX_PLUGIN_LIST_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC, name, base),     \
    /**/

#define HPX_PLUGIN_EXPORTER_NAME_DYNAMIC(name, base, cname)                   \
    HPX_PLUGIN_EXPORTER_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC, name, base,  \
        cname)                                                                \
    /**/

#define HPX_PLUGIN_EXPORTER_INSTANCE_NAME_DYNAMIC(name, base, cname)          \
    HPX_PLUGIN_EXPORTER_INSTANCE_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC,     \
        name, base, cname)                                                    \
    /**/

#define HPX_PLUGIN_FORCE_LOAD_NAME_DYNAMIC(name, base)                        \
    HPX_PLUGIN_FORCE_LOAD_NAME_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC, name, base)\
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_EXPORT_(prefix, name, BaseType, ActualType, actualname,    \
        classname)                                                            \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any> *      \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME_(prefix, name, classname)();      \
                                                                              \
    namespace {                                                               \
        struct HPX_PLUGIN_EXPORTER_NAME_(prefix, name, actualname, classname) \
        {                                                                     \
            HPX_PLUGIN_EXPORTER_NAME_(prefix, name, actualname, classname)()  \
            {                                                                 \
                static hpx::util::plugin::concrete_factory<                   \
                    BaseType, ActualType > cf;                                \
                hpx::util::plugin::abstract_factory<BaseType>* w = &cf;       \
                std::string actname(BOOST_PP_STRINGIZE(actualname));          \
                boost::algorithm::to_lower(actname);                          \
                HPX_PLUGIN_LIST_NAME_(prefix, name, classname)()->insert(     \
                    std::make_pair(actname, w));                              \
            }                                                                 \
        } HPX_PLUGIN_EXPORTER_INSTANCE_NAME_(prefix, name, actualname,        \
            classname);                                                       \
    }                                                                         \
    /**/

#define HPX_PLUGIN_EXPORT(name, BaseType, ActualType, actualname, classname)  \
    HPX_PLUGIN_EXPORT_(HPX_PLUGIN_SYMBOLS_PREFIX, name, BaseType, ActualType, \
        actualname, classname)                                                \
    /**/

#define HPX_PLUGIN_EXPORT_DYNAMIC(name, BaseType, ActualType, actualname,     \
        classname)                                                            \
    HPX_PLUGIN_EXPORT_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC, name, BaseType,     \
        ActualType, actualname, classname)                                    \
    /**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_PLUGIN_EXPORT_LIST_(prefix, name, classname)                      \
    extern "C" HPX_PLUGIN_EXPORT_API std::map<std::string, boost::any> *      \
        HPX_PLUGIN_API HPX_PLUGIN_LIST_NAME_(prefix, name, classname)()       \
    {                                                                         \
        static std::map<std::string, boost::any> r;                           \
        return &r;                                                            \
    }                                                                         \
    extern "C" HPX_PLUGIN_EXPORT_API                                          \
        void HPX_PLUGIN_FORCE_LOAD_NAME_(prefix, name, classname)()           \
    {                                                                         \
        HPX_PLUGIN_LIST_NAME_(prefix, name, classname)();                     \
    }                                                                         \
    /**/

#define HPX_PLUGIN_EXPORT_LIST(name, classname)                               \
    HPX_PLUGIN_EXPORT_LIST_(HPX_PLUGIN_SYMBOLS_PREFIX, name, classname)       \
    /**/

#define HPX_PLUGIN_EXPORT_LIST_DYNAMIC(name, classname)                       \
    HPX_PLUGIN_EXPORT_LIST_(HPX_PLUGIN_SYMBOLS_PREFIX_DYNAMIC, name,          \
        classname)                                                            \
    /**/

#endif

