//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/command_line_handling_local/late_command_line_handling_local.hpp>
#include <hpx/command_line_handling_local/parse_command_line_local.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/runtime_configuration.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace hpx { namespace local { namespace detail {
    void decode(std::string& str, char const* s, char const* r)
    {
        std::string::size_type pos = 0;
        while ((pos = str.find(s, pos)) != std::string::npos)
        {
            str.replace(pos, 2, r);
        }
    }

    std::string decode_string(std::string str)
    {
        decode(str, "\\n", "\n");
        return str;
    }

    int handle_late_commandline_options(util::runtime_configuration& ini,
        hpx::program_options::options_description const& options,
        void (*handle_print_bind)(std::size_t))
    {
        // do secondary command line processing, check validity of options only
        try
        {
            std::string unknown_cmd_line(
                ini.get_entry("hpx.unknown_cmd_line", ""));
            if (!unknown_cmd_line.empty())
            {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                hpx::program_options::variables_map vm;

                util::commandline_error_mode mode = util::rethrow_on_error;
                std::string allow_unknown(
                    ini.get_entry("hpx.commandline.allow_unknown", "0"));
                if (allow_unknown != "0")
                    mode = util::allow_unregistered;

                std::vector<std::string> still_unregistered_options;
                parse_commandline(ini, options, unknown_cmd_line, vm, mode,
                    nullptr, &still_unregistered_options);

                std::string still_unknown_commandline;
                for (std::size_t i = 1; i < still_unregistered_options.size();
                     ++i)
                {
                    if (i != 1)
                    {
                        still_unknown_commandline += " ";
                    }
                    still_unknown_commandline +=
                        util::detail::enquote(still_unregistered_options[i]);
                }

                if (!still_unknown_commandline.empty())
                {
                    util::section* s = ini.get_section("hpx");
                    HPX_ASSERT(s != nullptr);
                    s->add_entry(
                        "unknown_cmd_line_option", still_unknown_commandline);
                }
            }

            std::string fullhelp(ini.get_entry("hpx.cmd_line_help", ""));
            if (!fullhelp.empty())
            {
                std::string help_option(
                    ini.get_entry("hpx.cmd_line_help_option", ""));
                if (0 == std::string("full").find(help_option))
                {
                    std::cout << decode_string(fullhelp);
                    std::cout << options << std::endl;
                }
                else
                {
                    throw hpx::detail::command_line_error(
                        "unknown help option: " + help_option);
                }
                return 1;
            }

            // secondary command line handling, looking for --exit and other
            // options
            std::string cmd_line =
                ini.get_entry("hpx.commandline.command", "") + " " +
                ini.get_entry("hpx.commandline.prepend_options", "") +
                ini.get_entry("hpx.commandline.options", "") +
                ini.get_entry("hpx.commandline.config_options", "");

            if (!cmd_line.empty())
            {
                std::string runtime_mode(ini.get_entry("hpx.runtime_mode", ""));
                hpx::program_options::variables_map vm;

                parse_commandline(ini, options, cmd_line, vm,
                    util::allow_unregistered |
                        util::report_missing_config_file);

                if (vm.count("hpx:print-bind"))
                {
                    std::size_t num_threads =
                        hpx::util::from_string<std::size_t>(
                            ini.get_entry("hpx.os_threads", 1));
                    handle_print_bind(num_threads);
                }

                if (vm.count("hpx:exit"))
                {
                    return 1;
                }
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "handle_late_commandline_options: "
                      << "command line processing: " << e.what() << std::endl;
            return -1;
        }

        return 0;
    }
}}}    // namespace hpx::local::detail
