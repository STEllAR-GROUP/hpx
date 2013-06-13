////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/exception.hpp>

#if defined(BOOST_WINDOWS)
#  include <windows.h>
#elif defined(__linux) || defined(linux) || defined(__linux__)
#  include <unistd.h>
#  include <linux/limits.h>
#elif __APPLE__
#  include <mach-o/dyld.h>
#elif defined(__FreeBSD__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#  include <vector>
#endif

#include <boost/cstdint.hpp>
#include <hpx/util/plugin/dll.hpp>
#include <boost/filesystem/path.hpp>

namespace hpx { namespace util
{
    std::string find_prefix(
        std::string library
        )
    {
#if !defined(__ANDROID__) && !defined(ANDROID) && !defined(__APPLE__) && !defined(__MIC)
        try {
            error_code ec;
            hpx::util::plugin::dll dll(HPX_MAKE_DLL_STRING(library));

            d.load_library(ec);
            if (ec) return HPX_PREFIX;

            using boost::filesystem::path;

            std::string const prefix =
                path(dll.get_directory(ec)).parent_path().parent_path().string();

            if (ec || prefix.empty())
                return HPX_PREFIX;

            return prefix;
        }
        catch (std::logic_error const&) {
            ;   // just ignore loader problems
        }
#endif
        return HPX_PREFIX;
    }

    ///////////////////////////////////////////////////////////////////////////////
    std::string get_executable_prefix()
    {
        using boost::filesystem::path;
        path p(get_executable_filename());

        return p.parent_path().parent_path().string();
    }

    std::string get_executable_filename()
    {
        std::string r;

#if defined(BOOST_WINDOWS)
        char exe_path[MAX_PATH + 1] = { '\0' };
        if (!GetModuleFileName(NULL, exe_path, sizeof(exe_path)))
        {
            HPX_THROW_EXCEPTION(hpx::dynamic_link_failure,
                "get_executable_filename",
                "unable to find executable filename");
        }
        r = exe_path;

#elif defined(__linux) || defined(linux) || defined(__linux__)
        char exe_path[PATH_MAX + 1];
        ssize_t length = readlink("/proc/self/exe", exe_path, sizeof(exe_path));

        if (length == -1) 
        {
            HPX_THROW_EXCEPTION(hpx::dynamic_link_failure,
                "get_executable_filename",
                "unable to find executable filename, /proc may be unavailable");
        }

        exe_path[length] = '\0';
        r = exe_path;

#elif defined(__APPLE__)
        char exe_path[PATH_MAX + 1];
        boost::uint32_t len = sizeof(exe_path) / sizeof(exe_path[0]);

        if (0 != _NSGetExecutablePath(exe_path, &len))
        {
            HPX_THROW_EXCEPTION(hpx::dynamic_link_failure,
                "get_executable_filename",
                "unable to find executable filename");
        }

        exe_path[len] = '\0';
        r = exe_path;

#elif defined(__FreeBSD__)
        int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
        size_t cb = 0;
        sysctl(mib, 4, NULL, &cb, NULL, 0);
        if (cb)
        {
            std::vector<char> buf(cb);
            sysctl(mib, 4, &buf[0], &cb, NULL, 0);
            r = &buf[0];
        }

#else
#  error Unsupported platform
#endif

        return r;
    }
}}

