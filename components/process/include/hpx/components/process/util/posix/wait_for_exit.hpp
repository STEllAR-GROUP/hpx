// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/modules/errors.hpp>
#include <sys/types.h>
#include <sys/wait.h>

namespace hpx { namespace components { namespace process { namespace posix {

template <class Process>
inline int wait_for_exit(const Process &p)
{
    pid_t ret;
    int status = 0;
    do
    {
        ret = ::waitpid(p.pid, &status, 0);
    } while ((ret == -1 && errno == EINTR) || (ret != -1 && !WIFEXITED(status)));

    if (ret == -1)
    {
        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
            "process::wait_for_exit", "waitpid(2) failed");
    }
    return WEXITSTATUS(status);
}

template <class Process>
inline int wait_for_exit(const Process &p, hpx::error_code &ec)
{
    pid_t ret;
    int status;
    do
    {
        ret = ::waitpid(p.pid, &status, 0);
    } while ((ret == -1 && errno == EINTR) || (ret != -1 && !WIFEXITED(status)));

    if (ret == -1)
    {
        HPX_THROWS_IF(ec, hpx::error::invalid_status,
            "process::wait_for_exit", "waitpid(2) failed");
    }
    else
    {
        ec = hpx::make_success_code();
        return -1;
    }
    return WEXITSTATUS(status);
}

}}}}

#endif
