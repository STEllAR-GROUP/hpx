//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/command_line_handling_local/config/defines.hpp>

#if defined(HPX_COMMAND_LINE_HANDLING_HAVE_JSON_CONFIGURATION_FILES)

#include <hpx/command_line_handling_local/json_config_file.hpp>
#include <hpx/command_line_handling_local/parse_command_line_local.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace hpx::local::detail {

    template <typename T>
    void generate_option(std::vector<std::string>& options,
        std::string const& key, T const& value)
    {
        if (value.is_array())
        {
            for (auto it = value.begin(); it != value.end(); ++it)
            {
                generate_option(options, key, *it);
            }
            return;
        }

        if (value.is_object())
        {
            std::string const basekey(key + ":");
            for (auto it = value.begin(); it != value.end(); ++it)
            {
                generate_option(options, basekey + it.key(), it.value());
            }
            return;
        }

        std::stringstream str;
        if (value.is_boolean())
        {
            if (value.template get<bool>())
            {
                str << "--" << key;
                options.push_back(str.str());
            }
            return;
        }

        str << "--" << key << "=";
        if (value.is_string())
        {
            str << value.template get<std::string>();
        }
        else if (value.is_number_integer())
        {
            str << value.template get<int>();
        }
        else if (value.is_number_unsigned())
        {
            str << value.template get<unsigned int>();
        }
        else if (value.is_number_float())
        {
            str << value.template get<double>();
        }
        options.push_back(str.str());
    }

    std::vector<std::string> read_json_config_file_options(
        std::string const& filename, util::commandline_error_mode error_mode)
    {
        nlohmann::json j;

        {
            std::ifstream ifs(filename.c_str());
            if (!ifs.is_open())
            {
                if (as_bool(error_mode &
                        util::commandline_error_mode::
                            report_missing_config_file))
                {
                    std::cerr
                        << "read_json_config_file_options: command line "
                           "warning: command line options file not found ("
                        << filename << ")" << std::endl;
                }
                return {};
            }
            j = nlohmann::json::parse(ifs, nullptr, true, true);
        }

        std::vector<std::string> options;
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            generate_option(options, it.key(), it.value());
        }
        return options;
    }
}    // namespace hpx::local::detail

#endif
