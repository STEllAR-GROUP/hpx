//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <string>
#include <vector>

namespace hpx { namespace config_registry {
    struct module_config
    {
        std::string module_name;
        std::vector<std::string> config_entries;
    };

    std::vector<module_config> const& get_module_configs();
    void add_module_config(module_config const& config);

    struct add_module_config_helper
    {
        add_module_config_helper(module_config const& config);
    };
}}    // namespace hpx::config_registry
