// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/environ.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

class inherit_env : public initializer_base
{
public:
    template <class PosixExecutor>
    void on_fork_setup(PosixExecutor &e) const
    {
#if defined(__FreeBSD__)
        e.env = freebsd_environ;
#else
        e.env = environ;
#endif
    }
};

}

}}}}

#endif
