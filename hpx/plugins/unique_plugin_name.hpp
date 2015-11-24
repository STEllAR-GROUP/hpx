//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UNIQUE_PLUGIN_NAME_MAR_24_2013_245PM)
#define HPX_UNIQUE_PLUGIN_NAME_MAR_24_2013_245PM

#include <boost/config.hpp>

#include <boost/preprocessor/stringize.hpp>

namespace hpx { namespace plugins
{
    template <typename PluginType>
    struct unique_plugin_name
    {
        static_assert(sizeof(PluginType) == 0, "plugin name is not defined");
    };
}}

#define HPX_DEF_UNIQUE_PLUGIN_NAME(PluginType, name)                          \
    namespace hpx { namespace plugins                                         \
    {                                                                         \
        template <>                                                           \
        struct unique_plugin_name<PluginType >                                \
        {                                                                     \
            typedef char const* type;                                         \
                                                                              \
            static type call (void)                                           \
            {                                                                 \
                return BOOST_PP_STRINGIZE(name);                              \
            }                                                                 \
        };                                                                    \
    }}                                                                        \
/**/

#endif

