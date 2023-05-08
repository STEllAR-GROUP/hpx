//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime_local/config_entry.hpp>

#include <hpx/program_options/parsers.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[]);

//////////////////////////////////////////////////////////////////////////////
// Forwarding of hpx_main, if necessary. This has to be in a separate
// translation unit to ensure the linker can pick or ignore this function,
// depending on whether the main executable defines this symbol or not.
HPX_WEAK_SYMBOL int hpx_main()
{
    std::string cmdline(
        hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));

    using namespace hpx::program_options;
#if defined(HPX_WINDOWS)
    std::vector<std::string> args = split_winmain(cmdline);
#else
    std::vector<std::string> args = split_unix(cmdline);
#endif

    constexpr char hpx_prefix[] = "--hpx:";
    constexpr std::size_t hpx_prefix_len = std::size(hpx_prefix) - 1;

    constexpr char hpx_positional[] = "positional";
    constexpr std::size_t hpx_positional_len = std::size(hpx_positional) - 1;

    // Copy all arguments which are not hpx related to a temporary array
    std::vector<char*> argv(args.size() + 1);
    std::size_t argcount = 0;
    for (auto& argument : args)
    {
        if (0 != argument.compare(0, hpx_prefix_len, hpx_prefix))
        {
            argv[argcount++] = const_cast<char*>(argument.data());
        }
        else if (0 ==
            argument.compare(
                hpx_prefix_len, hpx_positional_len, hpx_positional))
        {
            std::string::size_type const p = argument.find_first_of('=');
            if (p != std::string::npos)
            {
                argument = argument.substr(p + 1);
                argv[argcount++] = const_cast<char*>(argument.data());
            }
        }
    }

    // add a single nullptr in the end as some application rely on that
    argv[argcount] = nullptr;

    // Invoke hpx_main
    return hpx_main(static_cast<int>(argcount), argv.data());
}
