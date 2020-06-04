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

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/modules/errors.hpp>

#include <hpx/components/process/util/posix/create_pipe.hpp>

#include <unistd.h>

namespace hpx { namespace components { namespace process { namespace posix
{
    pipe create_pipe()
    {
        int fds[2];
        if (::pipe(fds) == -1)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "posix::create_pipe", "pipe(2) failed");
        }
        return pipe(fds[0], fds[1]);
    }

    pipe create_pipe(hpx::error_code &ec)
    {
        int fds[2] = { 0, 0 };
        if (::pipe(fds) == -1)
        {
            HPX_THROWS_IF(ec, invalid_status,
                "posix::create_pipe", "pipe(2) failed");
        }
        else
        {
            ec = hpx::make_success_code();
        }
        return pipe(fds[0], fds[1]);
    }
}}}}

#endif
