//  Copyright (c) 2007-2012 Hartmut Kaiser
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
// Forwarding of hpx_main, if necessary. This has to be in a separate
// translation unit to ensure the linker can pick or ignore this function,
// depending on whether the main executable defines this symbol or not.
HPX_WEAK_SYMBOL int hpx_main()
{
    std::string cmdline(hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));

    using namespace hpx::program_options;
#if defined(HPX_WINDOWS)
    std::vector<std::string> args = split_winmain(cmdline);
#else
    std::vector<std::string> args = split_unix(cmdline);
#endif

    constexpr char const hpx_prefix[] = "--hpx:";
    constexpr char const hpx_prefix_len =
        (sizeof(hpx_prefix) / sizeof(hpx_prefix[0])) - 1;

    constexpr char const hpx_positional[] = "positional";
    constexpr char const hpx_positional_len =
        (sizeof(hpx_positional) / sizeof(hpx_positional[0])) - 1;

    // Copy all arguments which are not hpx related to a temporary array
    std::vector<char*> argv(args.size()+1);
    std::size_t argcount = 0;
    for (auto& arg : args)
    {
        if (0 != arg.compare(0, hpx_prefix_len, hpx_prefix))
        {
            argv[argcount++] = const_cast<char*>(arg.data());
        }
        else if (0 ==
            arg.compare(hpx_prefix_len, hpx_positional_len, hpx_positional))
        {
            std::string::size_type p = arg.find_first_of("=");
            if (p != std::string::npos) {
                arg = arg.substr(p+1);
                argv[argcount++] = const_cast<char*>(arg.data());
            }
        }
    }

    // add a single nullptr in the end as some application rely on that
    argv[argcount] = nullptr;

    // Invoke hpx_main
    return hpx_main(static_cast<int>(argcount), argv.data());
}
