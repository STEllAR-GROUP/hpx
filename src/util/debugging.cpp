//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/debugging.hpp>

#include <iostream>

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
    /* UNIX-style OS. ------------------------------------------- */
#include <unistd.h>
#endif

#if defined(HPX_WINDOWS)
#include <Windows.h>
#endif    // HPX_WINDOWS

#if defined(_POSIX_VERSION)
#include <boost/asio/ip/host_name.hpp>
#endif

namespace hpx {
namespace util {
    void attach_debugger()
    {
#if defined(_POSIX_VERSION)
        volatile int i = 0;
        std::cerr << "PID: " << getpid() << " on "
                  << boost::asio::ip::host_name()
                  << " ready for attaching debugger. Once attached set i = 1 "
                     "and continue"
                  << std::endl;
        while (i == 0)
        {
            sleep(1);
        }
#elif defined(HPX_WINDOWS)
        DebugBreak();
#endif
    }
}    // end namespace util
}    // end namespace hpx
