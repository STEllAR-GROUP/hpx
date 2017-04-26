//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/runtime_fwd.hpp>

#include <hpx/components/iostreams/ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace iostreams { namespace detail
{
    std::stringstream& get_consolestream()
    {
        static std::stringstream console_stream;
        return console_stream;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type return_id_type(future<bool> f, naming::id_type id)
    {
        f.get();        //re-throw any errors
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::id_type>
    create_ostream(char const* cout_name, std::ostream& strm)
    {
        LRT_(info) << "detail::create_ostream: creating '"
                   << cout_name << "' stream object";

        if (agas::is_console())
        {
            typedef components::component<server::output_stream> ostream_type;

            naming::id_type cout_id(
                components::server::construct<ostream_type>(std::ref(strm)),
                naming::id_type::managed);

            return agas::register_name(cout_name, cout_id).then(
                util::bind(&return_id_type, util::placeholders::_1, cout_id));
        }

        // the console locality will create the ostream during startup
        return agas::on_symbol_namespace_event(cout_name, true);
    }

    ///////////////////////////////////////////////////////////////////////////
//     void release_ostream(char const* name, naming::id_type const& id)
//     {
//         LRT_(info) << "detail::release_ostream: destroying '"
//                    << name << "' stream object";
//
//         if (agas::is_console())
//         {
//             // now unregister the object from AGAS
//             agas::unregister_name(launch::sync, name);
//         }
//     }
}}}

namespace hpx { namespace iostreams
{
    // force the creation of the singleton stream objects
    void create_cout()
    {
        if (!agas::is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::create_cout",
                "this function should be called on the console only");
        }
        detail::create_ostream(detail::cout_tag());
    }

    void create_cerr()
    {
        if (!agas::is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::create_cerr",
                "this function should be called on the console only");
        }
        detail::create_ostream(detail::cerr_tag());
    }

    void create_consolestream()
    {
        if (!agas::is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::create_consolestream",
                "this function should be called on the console only");
        }
        detail::create_ostream(detail::consolestream_tag());
    }

    std::stringstream const& get_consolestream()
    {
        if (get_runtime_ptr() != 0 && !agas::is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::get_consolestream",
                "this function should be called on the console only");
        }
        return detail::get_consolestream();
    }
}}

///////////////////////////////////////////////////////////////////////////////
HPX_PLAIN_ACTION(hpx::iostreams::create_cout, create_cout_action);
HPX_PLAIN_ACTION(hpx::iostreams::create_cerr, create_cerr_action);
HPX_PLAIN_ACTION(hpx::iostreams::create_consolestream, create_consolestream_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // global standard ostream objects
    iostreams::ostream<> cout;
    iostreams::ostream<> cerr;

    // extension: singleton stringstream on console
    iostreams::ostream<> consolestream;
    std::stringstream const& get_consolestream()
    {
        return iostreams::get_consolestream();
    }
}

