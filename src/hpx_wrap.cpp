//  Copyright (c) 20018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

// The following implementation has been divided for Linux and Mac OSX
#if (HPX_HAVE_DYNAMIC_HPX_MAIN != 0) && \
    (defined(__linux) || defined(__linux__) || defined(linux) || \
    defined(__APPLE__))

namespace hpx_start
{
    // include_libhpx_wrap is a weak symbol which helps to determine the course
    // of function calls at runtime. It has a default value of `false` which
    // corresponds to the program's entry point being main().
    // It is overridden in hpx/hpx_main.hpp. Thus, inclusion of the header file
    // will change the program's entry point to HPX's own custom entry point
    // initialize_main. Subsequent calls before entering main() are handled
    // by this code.
    extern bool include_libhpx_wrap;
    bool include_libhpx_wrap __attribute__((weak)) = false;
}

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <iostream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Function declarations
//
namespace hpx_start
{
    // Main entry point of HPX runtime system
    extern int hpx_entry(int argc, char* argv[]);
}

// Program's entry point depending upon Operating System.
// For Mac OSX it is the program's entry point. In case of Linux
// it is called by __wrap_main.
extern "C" int initialize_main(int argc, char** argv);

#if defined(__linux) || defined(__linux__) || defined(linux)
// Actual main() function
extern "C" int __real_main(int argc, char** argv);

// Our wrapper for main() function
extern "C" int __wrap_main(int argc, char** argv);
#endif

#if defined(__APPLE__)
// Declaration for main() for Mac OS implementation
extern "C" int main(int argc, char** argv);
#endif

namespace hpx_start
{
    // main entry point of the HPX runtime system
    int hpx_entry(int argc, char* argv[])
    {
        #if defined(__linux) || defined(__linux__) || defined(linux)
        // Call to the main() function
        int return_value = __real_main(argc, argv);
        #else /* APPLE */
        // call to the main() function
        int return_value = main(argc, argv);
        #endif

        //Finalizing the HPX runtime
        return hpx::finalize(return_value);
    }
}


// This is the main entry point of C runtime system.
// The HPX runtime system is initialized here, which
// is similar to initializing HPX from main() and utilising
// the hpx_main() as the entry point.
extern "C" int initialize_main(int argc, char** argv)
{
    #if defined(__APPLE__)
    if(hpx_start::include_libhpx_wrap)
    {
    #endif
        // Configuring HPX system before runtime
        std::vector<std::string> const cfg = {
            "hpx.commandline.allow_unknown!=1",
            "hpx.commandline.aliasing=0",
        };

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        hpx::util::function_nonser<int(int, char**)> start_function =
            hpx::util::bind(&hpx_start::hpx_entry, _1, _2);

        // Initialize the HPX runtime system
        return hpx::init(start_function, argc, argv,
            cfg, hpx::runtime_mode_console);
    #if defined(__APPLE__)
    }
    return main(argc, argv);
    #endif
}

#if defined(__linux) || defined(__linux__) || defined(linux)
////////////////////////////////////////////////////////////////////////////////
// Wrapper for main function
//

// We are wrapping the main function to initialize our
// runtime system prior to real main call.
extern "C" int __wrap_main(int argc, char** argv)
{
    // We determine the function call stack at runtime
    // from the value of include_libhpx_wrap. If hpx_main
    // is included include_libhpx_wrap is set to 1
    // due to override variable.
    if(hpx_start::include_libhpx_wrap)
        return initialize_main(argc, argv);

    // call main() since hpx_main.hpp is not included
    return __real_main(argc, argv);
}
////////////////////////////////////////////////////////////////////////////////
#endif

#endif
