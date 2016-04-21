//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0226PM)
#define HPX_COMPONENTS_CONSOLE_ERROR_SINK_JAN_23_2009_0226PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

#include <boost/exception_ptr.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    // console logging happens here
    void console_error_sink(boost::exception_ptr const&);

    typedef actions::action<
        void (*)(boost::exception_ptr const&), console_error_sink
    > console_error_sink_action;
}}}

HPX_ACTION_HAS_CRITICAL_PRIORITY(
    hpx::components::server::console_error_sink_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::console_error_sink_action,
    console_error_sink_action)

#endif
