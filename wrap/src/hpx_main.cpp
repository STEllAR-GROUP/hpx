//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx_user_main_config.hpp>

#include <string>
#include <vector>

namespace hpx_startup {

    namespace {

        std::vector<std::string> (*prev_user_main_config_function)(
            std::vector<std::string> const&) = nullptr;

        std::vector<std::string> enable_run_main(
            std::vector<std::string> const& config)
        {
            std::vector<std::string> cfg(config);
            cfg.emplace(cfg.begin(), "hpx.run_hpx_main!=1");
            if (prev_user_main_config_function)
                return prev_user_main_config_function(cfg);
            return cfg;
        }
    }    // namespace

    void install_user_main_config()
    {
        prev_user_main_config_function = hpx_startup::user_main_config_function;
        hpx_startup::user_main_config_function = &enable_run_main;
    }
}    // namespace hpx_startup
