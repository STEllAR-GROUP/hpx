//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/config_registry.hpp>

#include <vector>

namespace hpx::config_registry {

    namespace detail {

        [[nodiscard]] std::vector<module_config>& get_module_configs()
        {
            static std::vector<module_config> configs;
            return configs;
        }
    }    // namespace detail

    std::vector<module_config> const& get_module_configs()
    {
        return detail::get_module_configs();
    }

    void add_module_config(module_config const& config)
    {
        detail::get_module_configs().push_back(config);
    }

    add_module_config_helper::add_module_config_helper(
        module_config const& config)
    {
        add_module_config(config);
    }
}    // namespace hpx::config_registry
