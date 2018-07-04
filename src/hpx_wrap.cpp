//  Copyright (c) 20018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The following implementation is only possible for Linux systems.
#if defined(__linux) || defined(__linux__) || defined(linux)

namespace hpx {
    // include_libhpx_wrap is a weak symbol which helps to determine the course
    // of function calls at runtime. It has a deafult value of 0 which
    // corresponds to the program's entry point being main().
    // It is overriden in hpx/hpx_main.hpp. Thus, inclusion of the header file
    // will change the program's entry point to HPX's own custom entry point
    // initialize_main. Subsequent calls before entering main() are handled
    // by this code.
    extern int include_libhpx_wrap;
    int include_libhpx_wrap __attribute__((weak)) = 0;
}

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include<vector>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Function declarations
//

namespace hpx{
    // Main entry point of HPX runtime system
    extern int hpx_entry(int argc, char* argv[]);
}

// HPX's implented program's entry point
extern int initialize_main(int argc, char** argv, char** envp);

// Real libc function __libc_start_main. Function definition can be
// found in the glibc source in `glibc/csu/libc-start.c`
extern "C" int __real___libc_start_main (
                int (*main)(int, char**, char**), int argc, char * * ubp_av,
                void (*init) (void), void (*fini) (void),
                void (*rtld_fini) (void), void (* stack_end));

// Wrapper function for __libc_start_main
extern "C" int __wrap___libc_start_main (
                int (*main)(int, char**, char**), int argc, char * * ubp_av,
                void (*init) (void), void (*fini) (void),
                void (*rtld_fini) (void), void (* stack_end));


////////////////////////////////////////////////////////////////////////////////
// Global pointers

namespace hpx {
    // actual_main is responsible to store the pointer to the main()
    // function. POSIX implementation of main requires pointer to envp
    // so it is stored as well.
    int (*actual_main)(int, char**, char**) = nullptr;
    char** __envp = nullptr;
}

namespace hpx {

    // main entry point of the HPX runtime system
    int hpx_entry(int argc, char* argv[])
    {
        // Call to the main() function
        int return_value = hpx::actual_main(argc, argv, __envp);

        //Finalizing the HPX runtime
        return hpx::finalize(return_value);
    }
}

// This is the main entry point of C runtime system.
// The HPX runtime system is initialized here, which
// is similar to initializing HPX from main() and utilising
// the hpx_main() as the entry point.
int initialize_main(int argc, char** argv, char** envp)
{
    // initializing envp pointer to utilize when
    // calling the actual main.
    hpx::__envp = envp;

    // Configuring HPX system before runtime
    std::vector<std::string> const cfg = {
        "hpx.commandline.allow_unknown!=1",
        "hpx.commandline.aliasing=0",
    };

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    hpx::util::function_nonser<int(int, char**)> start_function =
        hpx::util::bind(&hpx::hpx_entry, _1, _2);

    // Initialize the HPX runtime system
    return hpx::init(start_function, argc, argv,
        cfg, hpx::runtime_mode_console);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper for the libc function __libc_start_main
//

// We are hooking into __libc_start_main to change the entry
// point of the C runtime system to our custom implemented
// function initialize_main
extern "C" int __wrap___libc_start_main (
                int (*main)(int, char**, char**), int argc, char * * ubp_av,
                void (*init) (void), void (*fini) (void),
                void (*rtld_fini) (void), void (* stack_end))
{

    // We determine the function call stack at runtime from the
    // value of include_libhpx_wrap.
    if(hpx::include_libhpx_wrap == 1) {
        // Assigning pointer to C main to actual_main
        hpx::actual_main = main;

        // Calling original __libc_start_main with our custom entry point.
        return __real___libc_start_main(&initialize_main, argc, ubp_av, init,
            fini, rtld_fini, stack_end);
    }
    return __real___libc_start_main(main, argc, ubp_av, init,
        fini, rtld_fini, stack_end);

}
////////////////////////////////////////////////////////////////////////////////

#endif
