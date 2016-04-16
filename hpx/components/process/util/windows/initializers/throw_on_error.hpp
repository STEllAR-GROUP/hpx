// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_WINDOWS_INITIALIZERS_THROW_ON_ERROR_HPP
#define HPX_PROCESS_WINDOWS_INITIALIZERS_THROW_ON_ERROR_HPP

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/throw_exception.hpp>
#include <hpx/components/process/util/windows/initializers/initializer_base.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace windows {

namespace initializers {

class throw_on_error : public initializer_base
{
public:
    template <class WindowsExecutor>
    void on_CreateProcess_error(WindowsExecutor&) const
    {
        HRESULT hr = GetLastError();
        LPVOID buffer = 0;
        if (!FormatMessage(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, hr,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
            (LPTSTR) &buffer, 0, NULL))
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "process::on_CreateProcess_error",
                boost::str(boost::format("format message failed with %x (while "
                    "retrieving message for %x)") % GetLastError() % hr));
            return;
        }

        std::string msg("CreateProcess() failed: ");
        msg += static_cast<char*>(buffer);
        LocalFree(buffer);
        HPX_THROW_EXCEPTION(kernel_error,
            "process::on_CreateProcess_error", msg);
    }
};

}

}}}}

#endif
#endif
