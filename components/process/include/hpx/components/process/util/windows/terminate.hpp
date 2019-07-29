// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_WINDOWS_TERMINATE_HPP
#define HPX_PROCESS_WINDOWS_TERMINATE_HPP

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>

#include <cstdlib>

namespace hpx { namespace components { namespace process { namespace windows {

template <class Process>
void terminate(const Process &p)
{
    if (!::TerminateProcess(p.process_handle(), EXIT_FAILURE))
    {
        HPX_THROW_EXCEPTION(invalid_status,
            "process::terminate", "TerminateProcess() failed");
    }
}

template <class Process>
void terminate(const Process &p, hpx::error_code &ec)
{
    if (!::TerminateProcess(p.process_handle(), EXIT_FAILURE))
    {
        HPX_THROWS_IF(ec, invalid_status,
            "process::terminate", "TerminateProcess() failed");
    }
    else
    {
        ec = hpx::make_success_code();
    }
}

}}}}

#endif
#endif
