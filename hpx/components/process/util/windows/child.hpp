// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_WINDOWS_CHILD_HPP
#define HPX_PROCESS_WINDOWS_CHILD_HPP

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/util/move.hpp>

#include <utility>
#include <cstring>

#include <windows.h>

namespace hpx { namespace components { namespace process { namespace windows {

class child
{
    HPX_MOVABLE_BUT_NOT_COPYABLE(child);

public:
    PROCESS_INFORMATION proc_info;

    child()
    {
        std::memset(&proc_info, '\0', sizeof(PROCESS_INFORMATION));
        proc_info.hProcess = INVALID_HANDLE_VALUE;
        proc_info.hThread = INVALID_HANDLE_VALUE;
    }

    explicit child(const PROCESS_INFORMATION &pi)
      : proc_info(pi)
    {}

    ~child()
    {
        if (proc_info.hProcess != INVALID_HANDLE_VALUE)
            ::CloseHandle(proc_info.hProcess);
        if (proc_info.hThread != INVALID_HANDLE_VALUE)
            ::CloseHandle(proc_info.hThread);
    }

    child(child && c) : proc_info(c.proc_info)
    {
        c.proc_info.hProcess = INVALID_HANDLE_VALUE;
        c.proc_info.hThread = INVALID_HANDLE_VALUE;
    }

    child &operator=(child && c)
    {
        if (proc_info.hProcess != INVALID_HANDLE_VALUE)
            ::CloseHandle(proc_info.hProcess);
        if (proc_info.hThread != INVALID_HANDLE_VALUE)
            ::CloseHandle(proc_info.hThread);

        proc_info = c.proc_info;
        c.proc_info.hProcess = INVALID_HANDLE_VALUE;
        c.proc_info.hThread = INVALID_HANDLE_VALUE;
        return *this;
    }

    HANDLE process_handle() const { return proc_info.hProcess; }
};

}}}}

#endif
#endif
