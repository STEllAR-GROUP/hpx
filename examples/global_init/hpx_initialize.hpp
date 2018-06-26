//  Copyright (c) 2018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

///////////////////////////////////////////////////////////////////////////////
// Store the command line arguments in global variables to make them available
// to the startup code.

#if defined(linux) || defined(__linux) || defined(__linux__)

int __argc = 0;
char** __argv = nullptr;

void set_argc_argv(int argc, char* argv[], char* env[])
{
    __argc = argc;
    __argv = argv;
}

__attribute__((section(".preinit_array")))
    void (*set_global_argc_argv)(int, char*[], char*[]) = &set_argc_argv;

#elif defined(__APPLE__)

#include <crt_externs.h>

inline int get_arraylen(char** argv)
{
    int count = 0;
    if (nullptr != argv)
    {
        while(nullptr != argv[count])
        ++count;
    }
    return count;
}

int __argc = get_arraylen(*_NSGetArgv());
char** __argv = *_NSGetArgv();

#endif

///////////////////////////////////////////////////////////////////////////////
// main function declaration
int main(int argc, char* argv[]);

///////////////////////////////////////////////////////////////////////////////
// This class demonstrates how to initialize the hpx runtime system with main
// running as an HPX thread.
//
struct manage_global_runtime {
    manage_global_runtime() {

        std::vector<std::string> const cfg = {
            // allow for unknown command line options
            "hpx.commandline.allow_unknown!=1",
            // disable HPX' short options
            "hpx.commandline.aliasing!=0"
        };

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        hpx::util::function_nonser<int(int, char**)> start_function =
            hpx::util::bind(&main, _1, _2);

        // Start main on an HPX thread and then suspend this thread until
        // hpx::finalize() is called
        hpx::init(start_function, __argc, __argv, cfg, hpx::runtime_mode_console);
    }

    ~manage_global_runtime() {
    // Destructor functions to go here
    }
};
// On object declaration the hpx runtime system is initialized
manage_global_runtime init;

///////////////////////////////////////////////////////////////////////////////
// This class is provides the necessary medium to de-initialize everything
// and then safely exit the program
//
struct destruct {
    destruct() {
			  // This will call the Destructors and then safely exit
        // Safely exit the program
        std::exit(EXIT_SUCCESS);
    }
};

destruct destroy;
///////////////////////////////////////////////////////////////////////////////
