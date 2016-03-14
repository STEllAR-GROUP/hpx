// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_INITIALIZER_BASE_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_INITIALIZER_BASE_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

struct initializer_base
{
    template <class PosixExecutor>
    void on_fork_setup(PosixExecutor&) const {}

    template <class PosixExecutor>
    void on_fork_error(PosixExecutor&) const {}

    template <class PosixExecutor>
    void on_fork_success(PosixExecutor&) const {}

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor&) const {}

    template <class PosixExecutor>
    void on_exec_error(PosixExecutor&) const {}
};

}

}}}}

#endif
#endif
