//  Copyright (c) 2005-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/get_commandline_args.hpp>

#ifdef __APPLE__

#include <stdio.h>
#include <crt_externs.h>
#include <mach-o/dyld.h>

#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__AIX__)

#include <hpx/util/stringstream.hpp>

#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>

#endif

namespace hpx { namespace util
{
#if defined(linux) || defined(__linux) || defined(__linux__)

    bool linux_get_cmd_line (std::vector<std::string>& cmdline)
    {
        std::string elem;
        hpx::util::osstream procfile;

        procfile << "/proc/" << getpid () << "/cmdline";
        std::ifstream proc(osstream_get_string(procfile).c_str(), std::ios::binary);
        if (!proc.is_open())
            return false;       // proc fs does not exist on this machine

        while (!proc.eof())
        {
            char c = proc.get();
            if (c == '\0')
            {
                cmdline.push_back(elem);
                elem.clear();
            }
            else
            {
                elem += c;
            }
        }

        if (!elem.empty())
            cmdline.push_back(elem);

        return true;
    }

    bool linux_get_args(std::vector<std::string>& args)
    {
        std::vector <std::string> cmdline;
        if (!linux_get_cmd_line(cmdline))
            return false;

        if (!cmdline.empty())
            std::copy(cmdline.begin(), cmdline.end(), std::back_inserter(args));

        return true;
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    //  Figure out the size of the given array of pointers
    inline int get_arraylen(char **array)
    {
        int count = 0;
        if (NULL != array) {
            while(NULL != array[count])
                ++count;   // simply count the strings
        }
        return count;
    }

    std::vector<std::string> get_commandline_args()
    {
        std::vector<std::string> args;
#if defined(BOOST_WINDOWS)
        // we need to calculate the size of argv, because Boost.Test just
        // NULL's out its own command line arguments, leaving us with
        // invalid command line argument entries
        int len = get_arraylen(__argv);
        std::copy(&__argv[0], &__argv[len], std::back_inserter(args));

#elif defined(linux) || defined(__linux) || defined(__linux__)
        // get args from /proc/$pid/cmdline
        if (!linux_get_args(args))
        {
            HPX_THROW_EXCEPTION(hpx::not_implemented,
                "Unable to extract arguments for this job on this platform.",
                "hpx::util::get_commandline_args");
        }

#elif defined(__AIX__)

#elif defined(__APPLE__)
        // we need to calculate the size of argv, because Boost.Test just
        // NULL's out its own command line arguments, leaving us with
        // invalid command line argument entries
        char **__argv = *_NSGetArgv();
        int len = get_arraylen(__argv);
        std::copy(&__argv[0], &__argv[len], std::back_inserter(args));
#else
#error "Don't know, how to access the executable arguments on this platform"
#endif
        return args;
    }
}}

