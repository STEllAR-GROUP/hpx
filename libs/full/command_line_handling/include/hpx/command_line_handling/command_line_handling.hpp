//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/modules/command_line_handling_local.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/modules/util.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT command_line_handling
      : hpx::local::detail::command_line_handling
    {
        using base_type = hpx::local::detail::command_line_handling;

        command_line_handling(runtime_configuration rtcfg,
            std::vector<std::string> ini_config,
            hpx::function<int(hpx::program_options::variables_map& vm)>
                hpx_main_f);

        int call(hpx::program_options::options_description const& desc_cmdline,
            int argc, char** argv,
            std::vector<std::shared_ptr<components::component_registry_base>>&
                component_registries);

        std::size_t node_;
        std::size_t num_localities_;

    protected:
        bool handle_arguments(util::manage_config& cfgmap,
            hpx::program_options::variables_map& vm,
            std::vector<std::string>& ini_config, std::size_t& node,
            bool initial = false);

        void enable_logging_settings(hpx::program_options::variables_map& vm,
            std::vector<std::string>& ini_config);
    };
}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
