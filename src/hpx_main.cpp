//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/config_entry.hpp>

#include <boost/program_options/parsers.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Forwarding of hpx_main, if necessary. This has to be in a separate
// translation unit to ensure the linker can pick or ignore this function,
// depending on whether the main executable defines this symbol or not.
int hpx_main()
{
    std::string cmdline(hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));

    using namespace boost::program_options;
#if defined(HPX_WINDOWS)
    std::vector<std::string> args = split_winmain(cmdline);
#else
    std::vector<std::string> args = split_unix(cmdline);
#endif

    // Copy all arguments which are not hpx related to a temporary array
    std::vector<char*> argv(args.size()+1);
    std::size_t argcount = 0;
    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (0 != args[i].find("--hpx:")) {
            argv[argcount++] = const_cast<char*>(args[i].data());
        }
        else if (6 == args[i].find("positional", 6)) {
            std::string::size_type p = args[i].find_first_of("=");
            if (p != std::string::npos) {
                args[i] = args[i].substr(p+1);
                argv[argcount++] = const_cast<char*>(args[i].data());
            }
        }
    }

    // add a single nullptr in the end as some application rely on that
    argv[argcount] = nullptr;

    // Invoke hpx_main
    return hpx_main(static_cast<int>(argcount), argv.data());
}
