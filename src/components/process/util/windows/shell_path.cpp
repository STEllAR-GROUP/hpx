// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/exception.hpp>
#include <hpx/components/process/util/windows/shell_path.hpp>

#include <boost/filesystem/path.hpp>

namespace hpx { namespace components { namespace process { namespace windows
{
    boost::filesystem::path shell_path()
    {
        TCHAR sysdir[MAX_PATH];
        UINT size = ::GetSystemDirectory(sysdir, sizeof(sysdir));
        if (!size)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "process::shell_path",
                "GetSystemDirectory() failed");
        }
        boost::filesystem::path p = sysdir;
        return p / "cmd.exe";
    }

    boost::filesystem::path shell_path(hpx::error_code &ec)
    {
        TCHAR sysdir[MAX_PATH];
        UINT size = ::GetSystemDirectory(sysdir, sizeof(sysdir));
        boost::filesystem::path p;
        if (!size)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "process::shell_path",
                "GetSystemDirectory() failed");
        }
        else
        {
            ec = hpx::make_success_code();
            p = sysdir;
            p /= "cmd.exe";
        }
        return p;
    }
}}}}

#endif
