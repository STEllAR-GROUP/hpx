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

#include <signal.h>

namespace hpx { namespace components { namespace process { namespace posix {

    template <class Process>
    void terminate(const Process& p)
    {
        if (::kill(p.pid, SIGKILL) == -1)
        {
            HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                "process::terminate", "kill(2) failed");
        }
    }

    template <class Process>
    void terminate(const Process& p, hpx::error_code& ec)
    {
        if (::kill(p.pid, SIGKILL) == -1)
        {
            HPX_THROWS_IF(ec, hpx::error::invalid_status, "process::terminate",
                "kill(2) failed");
        }
        else
        {
            ec = hpx::make_success_code();
        }
    }

}}}}    // namespace hpx::components::process::posix

#endif
