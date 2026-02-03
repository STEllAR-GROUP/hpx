//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2010 Daniel James
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// A few methods for getting and manipulating file handles.

#pragma once

#include <hpx/config.hpp>

#include <string>

#if defined(HPX_WINDOWS)
#include <io.h>    // low-level file i/o.
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#endif

namespace hpx::iostream::test {

#if defined(HPX_WINDOWS)

    // Windows implementation
    hpx::iostream::file_handle open_file_handle(std::string const& name)
    {
        HANDLE handle = ::CreateFileA(name.c_str(),
            GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

        HPX_TEST(handle != INVALID_HANDLE_VALUE);
        return handle;
    }

    void close_file_handle(hpx::iostream::file_handle handle)
    {
        HPX_TEST(::CloseHandle(handle) == 1);
    }

    constexpr void check_handle_open(hpx::iostream::file_handle) noexcept {}
    constexpr void check_handle_closed(hpx::iostream::file_handle) noexcept {}
#else

    // Non-windows implementation
    hpx::iostream::file_handle open_file_handle(std::string const& name)
    {
        int oflag = O_RDWR;

#ifdef _LARGEFILE64_SOURCE
        oflag |= O_LARGEFILE;
#endif

        // Calculate pmode argument to open.
        mode_t pmode =
            S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

        // Open file.
        int fd = ::open(name.c_str(), oflag, pmode);
        HPX_TEST(fd != -1);

        return fd;
    }

    void close_file_handle(hpx::iostream::file_handle handle)
    {
        HPX_TEST(::close(handle) != -1);
    }

    // This is pretty dubious. First you must make sure that no other
    // operations that could open a descriptor are called before this
    // check, otherwise it's quite likely that a closed descriptor
    // could be used. Secondly, I'm not sure if it's guaranteed that
    // fcntl will know that the descriptor is closed but this seems
    // to work okay, and I can't see any other convenient way to check
    // that a descripter has been closed.
    bool is_handle_open(hpx::iostream::file_handle handle)
    {
        return ::fcntl(handle, F_GETFD) != -1;
    }

    inline void check_handle_open(hpx::iostream::file_handle handle)
    {
        HPX_TEST(::hpx::iostream::test::is_handle_open(handle));
    }

    inline void check_handle_closed(hpx::iostream::file_handle handle)
    {
        HPX_TEST(!::hpx::iostream::test::is_handle_open(handle));
    }
#endif
}    // namespace hpx::iostream::test
