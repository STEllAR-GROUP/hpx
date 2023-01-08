//  Copyright (c) 2017-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This module needs an empty config-registry entry for testing purposes

#include <hpx/modules/config_registry.hpp>

namespace hpx::config_registry_cfg {

    config_registry::add_module_config_helper add_config{
        {"config_registry", {}}};

}    // namespace hpx::config_registry_cfg
