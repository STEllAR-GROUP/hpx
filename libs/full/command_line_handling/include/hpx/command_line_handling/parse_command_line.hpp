//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/command_line_handling_local.hpp>
#include <hpx/modules/ini.hpp>
#include <hpx/modules/runtime_configuration.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    HPX_CXX_EXPORT HPX_EXPORT bool parse_commandline(
        hpx::util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& cmdline, hpx::program_options::variables_map& vm,
        std::size_t node,
        commandline_error_mode error_mode =
            commandline_error_mode::return_on_error,
        hpx::runtime_mode mode = runtime_mode::default_,
        hpx::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    HPX_CXX_EXPORT HPX_EXPORT bool parse_commandline(
        hpx::util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& arg0, std::vector<std::string> const& args,
        hpx::program_options::variables_map& vm, std::size_t node,
        commandline_error_mode error_mode =
            commandline_error_mode::return_on_error,
        hpx::runtime_mode mode = runtime_mode::default_,
        hpx::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);
}    // namespace hpx::util
