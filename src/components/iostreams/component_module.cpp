//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/component_startup_shutdown.hpp>
#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/util/function.hpp>

#include <hpx/components/iostreams/server/output_stream.hpp>
#include <hpx/components/iostreams/ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::iostreams::server::output_stream ostream_type;

HPX_REGISTER_COMPONENT(
    hpx::components::component<ostream_type>,
    output_stream_factory, hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(ostream_type)

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_ID(
    ostream_type::write_async_action,
    output_stream_write_async_action,
    hpx::actions::output_stream_write_async_action_id)

HPX_REGISTER_ACTION_ID(
    ostream_type::write_sync_action,
    output_stream_write_sync_action,
    hpx::actions::output_stream_write_sync_action_id)

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
namespace hpx { namespace iostreams { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    void register_ostreams()
    {
        hpx::cout.initialize(iostreams::detail::cout_tag());
        hpx::cerr.initialize(iostreams::detail::cerr_tag());
        hpx::consolestream.initialize(iostreams::detail::consolestream_tag());
    }

    void unregister_ostreams()
    {
        hpx::cout.uninitialize();
        hpx::cerr.uninitialize();
        hpx::consolestream.uninitialize();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(startup_function_type& startup_func, bool& pre_startup)
    {
        // return our startup-function
        startup_func = register_ostreams;   // function to run during startup
        pre_startup = true;                 // run as pre-startup function
        return true;
    }

    bool get_shutdown(shutdown_function_type& shutdown_func, bool& pre_shutdown)
    {
        // return our startup-function
        shutdown_func = unregister_ostreams;   // function to run during startup
        pre_shutdown = false;                 // run as pre-startup function
        return true;
    }
}}}

// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(
    hpx::iostreams::detail::get_startup, hpx::iostreams::detail::get_shutdown);

