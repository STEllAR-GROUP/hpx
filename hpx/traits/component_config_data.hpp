//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_COMPONENT_CONFIG_DATA_MAR_25_2013_0810AM)
#define HPX_TRAITS_COMPONENT_CONFIG_DATA_MAR_25_2013_0810AM

#include <hpx/config.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for component config data injection
    template <typename Component, typename Enable = void>
    struct component_config_data
    {
        // by default no additional config data is injected into the factory
        static char const* call()
        {
            return 0;
        }
    };
}}

#endif

