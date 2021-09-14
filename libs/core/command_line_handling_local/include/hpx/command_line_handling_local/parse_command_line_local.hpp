//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/ini/ini.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx {
    namespace util {
        enum commandline_error_mode
        {
            return_on_error,
            rethrow_on_error,
            allow_unregistered,
            ignore_aliases = 0x40,
            report_missing_config_file = 0x80
        };

        namespace detail {
            inline std::string enquote(std::string const& arg)
            {
                if (arg.find_first_of(" \t\"") != std::string::npos)
                    return std::string("\"") + arg + "\"";
                return arg;
            }
        }    // namespace detail
    }        // namespace util

    namespace local { namespace detail {
        HPX_CORE_EXPORT bool parse_commandline(hpx::util::section const& rtcfg,
            hpx::program_options::options_description const& app_options,
            std::string const& cmdline, hpx::program_options::variables_map& vm,
            int error_mode = util::return_on_error,
            hpx::program_options::options_description* visible = nullptr,
            std::vector<std::string>* unregistered_options = nullptr);

        HPX_CORE_EXPORT bool parse_commandline(hpx::util::section const& rtcfg,
            hpx::program_options::options_description const& app_options,
            std::string const& arg0, std::vector<std::string> const& args,
            hpx::program_options::variables_map& vm,
            int error_mode = util::return_on_error,
            hpx::program_options::options_description* visible = nullptr,
            std::vector<std::string>* unregistered_options = nullptr);

        HPX_CORE_EXPORT std::string reconstruct_command_line(
            hpx::program_options::variables_map const& vm);
    }}    // namespace local::detail
}    // namespace hpx
