//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0621PM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    // Stub function which applies the console_error_sink action.
    HPX_EXPORT void console_error_sink(naming::id_type const& dst,
        boost::exception_ptr const& e);

    // Stub function which applies the console_error_sink action.
    HPX_EXPORT void console_error_sink(boost::exception_ptr const& e);
}}

#endif

