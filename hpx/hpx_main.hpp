//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_HPX_MAIN_HPP
#define HPX_HPX_MAIN_HPP

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

#if (defined(__GNUC__) || (__clang__)) && !defined(HPX_HAVE_STATIC_LINKING)
#define _GNU_SOURCE

#include <hpx/hpx.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/hpx_start.hpp>

#include <dlfcn.h>
#include <iostream>
#include <vector>

// To store the pointer to the compiler's main
int (*actual_main)(int, char**, char**);
int __argc = 0;
char** __argv = nullptr;
char** __envp = nullptr;


namespace hpx {
    ///////////////////////////////////////////////////////////////////////////////
    // Default implementation of hpx_entry() for initializing the HPX runtime
    // system
    //
    int hpx_entry(int argc, char* argv[])
    {
        // Call to the Compiler's main with HPX runt time
        // initiated.
        actual_main(__argc, __argv, __envp);

        return hpx::finalize();
    }

}

///////////////////////////////////////////////////////////////////////////////
// Default entry point for the runtime system.
//
// Note: This function is now the equivalent of Comipler's main i.e the entry
// point has been shifted from "main" to "initializing_main"
//
int __initializing_main (int argc, char** argv, char** envp)
{
    __argc = argc;
    __argv = argv;
    __envp = envp;

    std::vector<std::string> const cfg = {
        // allow for unknown command line options
        "hpx.commandline.allow_unknown!=1",
    };

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    hpx::util::function_nonser<int(int, char**)> start_function =
        hpx::util::bind(&hpx::hpx_entry, _1, _2);

    // Initializing HPX runtime using hpx::init
    return hpx::init(start_function, argc, argv, cfg, hpx::runtime_mode_console);
}

// Wrapper for the Compiler's __libc_start_main function.
extern "C" int __libc_start_main (
                int (*main)(int, char**, char**), int argc, char * * ubp_av,
                void (*init) (void), void (*fini) (void),
                void (*rtld_fini) (void), void (* stack_end))
{
    // Storing pointer to the __libc_start_main
    int (*real_start_main)(int (*main) (int, char**, char**), int argc,
        char** ubp_av, void (*init) (void),
        void (*fini) (void), void (*rtld_fini) (void), void (* stack_end))
        =
        (int (*)(int (*)(int, char**, char**), int, char**, void (*)(),
        void (*)(), void (*)(), void*))dlsym(RTLD_NEXT, "__libc_start_main");

    actual_main = main;

    // call original __libc_start_main, but replace "main" with our custom implementation
    return real_start_main(__initializing_main, argc, ubp_av,
            init, fini, rtld_fini, stack_end);
}

#else
#include <hpx/hpx_main_impl.hpp>

// We support redefining the plain C-main provided by the user to be executed
// as the first HPX-thread (equivalent to hpx_main()). This is implemented by
// a macro redefining main, so we disable it by default.
#define main hpx_startup::user_main
#endif

#endif /*HPX_HPX_MAIN_HPP*/
