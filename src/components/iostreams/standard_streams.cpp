//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/components/iostreams/ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace iostreams { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    std::ostream& get_outstream(cout_tag)
    {
        return std::cout;
    }

    std::ostream& get_outstream(cerr_tag)
    {
        return std::cerr;
    }

    char const* const cout_name = "/locality#console/output_stream#cout";
    char const* const cerr_name = "/locality#console/output_stream#cerr";

    char const* const get_outstream_name(cout_tag)
    {
        return cout_name;
    }

    char const* const get_outstream_name(cerr_tag)
    {
        return cerr_name;
    }

    ///////////////////////////////////////////////////////////////////////////
    naming::id_type return_id_type(future<bool> f, naming::id_type id)
    {
        f.get();        //re-throw any errors
        return id;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag>
    hpx::future<naming::id_type> create_ostream(Tag tag)
    {
        LRT_(info) << "detail::create_ostream: creating '"
                   << cout_name << "' stream object";

        char const* cout_name = get_outstream_name(tag);
        naming::resolver_client& agas_client = get_runtime().get_agas_client();
        if (agas_client.is_console())
        {
            typedef components::managed_component<server::output_stream>
                ostream_type;

            naming::id_type cout_id(
                components::server::create_with_args<ostream_type>(
                    boost::ref(detail::get_outstream(tag))),
                naming::id_type::managed);

            return agas::register_name(cout_name, cout_id).then(
                util::bind(&return_id_type, util::placeholders::_1, cout_id));
        }

        // the console locality will create the ostream during startup
        return agas::on_symbol_namespace_event(cout_name, agas::symbol_ns_bind, true);
    }
}}}

namespace hpx { namespace iostreams
{
    // force the creation of the singleton stream objects
    void create_cout()
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();
        if (!agas_client.is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::create_cout",
                "this function should be called on the console only");
        }
        detail::create_ostream(detail::cout_tag());
    }

    void create_cerr()
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();
        if (!agas_client.is_console())
        {
            HPX_THROW_EXCEPTION(service_unavailable,
                "hpx::iostreams::create_cerr",
                "this function should be called on the console only");
        }
        detail::create_ostream(detail::cerr_tag());
    }
}}

///////////////////////////////////////////////////////////////////////////////
HPX_PLAIN_ACTION(hpx::iostreams::create_cout, create_cout_action);
HPX_PLAIN_ACTION(hpx::iostreams::create_cerr, create_cerr_action);

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    // global standard ostream objects
    iostreams::ostream cout;
    iostreams::ostream cerr;
}

