//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/runtime_configuration/ini.hpp>
#include <hpx/runtime_configuration/runtime_mode.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    enum commandline_error_mode
    {
        return_on_error,
        rethrow_on_error,
        allow_unregistered,
        report_missing_config_file = 0x80
    };

    ///////////////////////////////////////////////////////////////////////////
    // parse the command line
    HPX_EXPORT bool parse_commandline(hpx::util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& cmdline, hpx::program_options::variables_map& vm,
        std::size_t node, int error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode::default_,
        hpx::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    HPX_EXPORT bool parse_commandline(hpx::util::section const& rtcfg,
        hpx::program_options::options_description const& app_options,
        std::string const& arg0, std::vector<std::string> const& args,
        hpx::program_options::variables_map& vm, std::size_t node,
        int error_mode = return_on_error,
        hpx::runtime_mode mode = runtime_mode::default_,
        hpx::program_options::options_description* visible = nullptr,
        std::vector<std::string>* unregistered_options = nullptr);

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT std::string reconstruct_command_line(
        hpx::program_options::variables_map const& vm);

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        inline std::string enquote(std::string const& arg)
        {
            if (arg.find_first_of(" \t\"") != std::string::npos)
                return std::string("\"") + arg + "\"";
            return arg;
        }
    }    // namespace detail
}}       // namespace hpx::util
