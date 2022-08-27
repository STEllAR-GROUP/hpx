//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2013-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/preprocessor.hpp>

namespace hpx::plugins {

    template <typename PluginType>
    struct unique_plugin_name
    {
        static_assert(sizeof(PluginType) == 0, "plugin name is not defined");
    };
}    // namespace hpx::plugins

#define HPX_DEF_UNIQUE_PLUGIN_NAME(PluginType, name)                           \
    namespace hpx { namespace plugins {                                        \
            template <>                                                        \
            struct unique_plugin_name<PluginType>                              \
            {                                                                  \
                using type = char const*;                                      \
                                                                               \
                static constexpr type call(void) noexcept                      \
                {                                                              \
                    return HPX_PP_STRINGIZE(name);                             \
                }                                                              \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/
