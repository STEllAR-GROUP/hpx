//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/throw_exception.hpp>
//#include <hpx/runtime/get_config_entry.hpp>

//#include <cstddef>
//#include <string>
//#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Forwarding of hpx_startup::user_main, if necessary. This has to be in a
// separate translation unit to ensure the linker can pick or ignore this function,
// depending on whether the main executable defines this symbol or not.
int hpx_startup::user_main()
{
//     std::string cmdline(hpx::get_config_entry("hpx.reconstructed_cmd_line", ""));
//
//     using namespace boost::program_options;
// #if defined(HPX_WINDOWS)
//     std::vector<std::string> args = split_winmain(cmdline);
// #else
//     std::vector<std::string> args = split_unix(cmdline);
// #endif
//
//     // Copy all arguments which are not hpx related to a temporary array
//     std::vector<char*> argv(args.size());
//     std::size_t argcount = 0;
//     for(std::size_t i = 0; i < args.size(); ++i)
//     {
//         if (0 != args[i].find("--hpx:"))
//             argv[argcount++] = const_cast<char*>(args[i].data());
//     }
//
//     // Invoke hpx_main
//     return user_main(static_cast<int>(argcount), argv.data());
    HPX_THROW_EXCEPTION(hpx::not_implemented,
        "The console locality does not implement any main entry point usable "
        "as the main HPX thread (e.g. no hpx_main, hpx_startup::user_main, etc.)",
        "hpx_startup::user_main");
}
