//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <exception>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    // Stub function which applies the console_error_sink action.
    HPX_EXPORT void console_error_sink(naming::id_type const& dst,
        std::exception_ptr const& e);

    // Stub function which applies the console_error_sink action.
    HPX_EXPORT void console_error_sink(std::exception_ptr const& e);
}}

#endif

