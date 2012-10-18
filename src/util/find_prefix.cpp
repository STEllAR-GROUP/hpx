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
#elif defined(__linux__)
#  include <unistd.h>
#  include <linux/limits.h>
#elif __APPLE__
#  include <mach-o/dyld.h>
#endif

#include <boost/cstdint.hpp>
#include <boost/plugin/dll.hpp>
#include <boost/filesystem/path.hpp>

namespace hpx { namespace util
{
    std::string find_prefix(
        std::string library
        )
    {
#if !defined(__ANDROID__) && !defined(ANDROID)
        try {
            boost::plugin::dll dll(HPX_MAKE_DLL_STRING(library));

            using boost::filesystem::path;

            std::string const prefix =
                path(dll.get_directory()).parent_path().parent_path().string();

            if (prefix.empty())
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

#elif defined(__linux__)
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
        char exe_path[MAXPATHLEN + 1];
        boost::uint32_t buf_length

        int length = _NSGetExecutablePath(exe_path, &len);
        if (0 != _NSGetExecutablePath(exe_path, &len))
        {
            HPX_THROW_EXCEPTION(hpx::dynamic_link_failure,
                "get_executable_filename",
                "unable to find executable filename");
        }

        exe_path[length] = '\0';
        r = exe_path;

#else
#  error Unsupported platform
#endif

        return r;
    }
}}

