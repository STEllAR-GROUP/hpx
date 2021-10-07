//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for plugin config data injection
    template <typename Plugin, typename Enable = void>
    struct plugin_config_data
    {
        // by default no additional config data is injected into the factory
        static char const* call()
        {
            return nullptr;
        }
    };
}}    // namespace hpx::traits
