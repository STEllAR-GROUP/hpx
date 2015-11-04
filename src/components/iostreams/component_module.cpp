//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/components/iostreams/server/output_stream.hpp>
#include <hpx/components/iostreams/ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::iostreams::server::output_stream ostream_type;

HPX_REGISTER_COMPONENT(
    hpx::components::managed_component<ostream_type>,
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
    bool get_startup(util::function_nonser<void()>& startup_func, bool& pre_startup)
    {
        // return our startup-function
        startup_func = register_ostreams;   // function to run during startup
        pre_startup = true;                 // run as pre-startup function
        return true;
    }

    bool get_shutdown(util::function_nonser<void()>& shutdown_func, bool& pre_shutdown)
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

