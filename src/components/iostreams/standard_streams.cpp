////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#if HPX_AGAS_VERSION > 0x10

#include <hpx/hpx.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/util/static.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

typedef hpx::actions::plain_action0<hpx::iostreams::create_cout>
    create_cout_action;

HPX_REGISTER_PLAIN_ACTION(create_cout_action);

typedef hpx::actions::plain_action0<hpx::iostreams::create_cerr>
    create_cerr_action;

HPX_REGISTER_PLAIN_ACTION(create_cerr_action);

namespace hpx { namespace iostreams
{

struct cout_raii
{
    typedef components::managed_component<server::output_stream> ostream_type;

    struct tag {};

    cout_raii()
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();

        if (agas_client.is_console())
        {
            client.reset(new lazy_ostream(naming::id_type
                ( components::server::create_one<ostream_type>
                    (boost::ref(std::cout))
                , naming::id_type::managed)));
            agas_client.registerid
                ( "/locality(console)/output_stream(cout)"
                , naming::strip_credit_from_gid(client->get_raw_gid())); 
        }

        else
        {
            naming::gid_type gid;

            if (!agas_client.queryid("/locality(console)/output_stream(cout)"
                                   , gid))
            {
                naming::gid_type console;

                agas_client.get_console_prefix(console);

                if (HPX_UNLIKELY(!console))
                {
                    HPX_THROW_EXCEPTION(no_registered_console,
                        "cout_raii::cout_raii",
                        "couldn't contact console");
                }

                lcos::eager_future<create_cout_action>(console).get();

                bool r = agas_client.queryid
                    ("/locality(console)/output_stream(cout)", gid);

                if (HPX_UNLIKELY(!r || !gid))
                {
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "cout_raii::cout_raii",
                        "couldn't create cout stream on the console locality");
                } 
            }

            client.reset(new lazy_ostream(naming::id_type
                (gid, naming::id_type::unmanaged))); 
        }
    }

    boost::shared_ptr<lazy_ostream> client;
};

lazy_ostream& cout()
{
    util::static_<cout_raii, cout_raii::tag> cout_;
    return *cout_.get().client;  
}

void create_cout()
{ cout(); }

struct cerr_raii
{
    typedef components::managed_component<server::output_stream> ostream_type;

    struct tag {};

    cerr_raii()
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();

        if (agas_client.is_console())
        {
            client.reset(new lazy_ostream(naming::id_type
                ( components::server::create_one<ostream_type>
                    (boost::ref(std::cerr))
                , naming::id_type::managed)));
            agas_client.registerid
                ( "/locality(console)/output_stream(cerr)"
                , naming::strip_credit_from_gid(client->get_raw_gid())); 
        }

        else
        {
            naming::gid_type gid;

            if (!agas_client.queryid("/locality(console)/output_stream(cerr)"
                                   , gid))
            {
                naming::gid_type console;

                agas_client.get_console_prefix(console);

                if (HPX_UNLIKELY(!console))
                {
                    HPX_THROW_EXCEPTION(no_registered_console,
                        "cerr_raii::cerr_raii",
                        "couldn't contact console");
                }

                lcos::eager_future<create_cerr_action>(console).get();

                bool r = agas_client.queryid
                    ("/locality(console)/output_stream(cerr)", gid);

                if (HPX_UNLIKELY(!r || !gid))
                {
                    HPX_THROW_EXCEPTION(service_unavailable,
                        "cerr_raii::cerr_raii",
                        "couldn't create cerr stream on the console locality");
                } 
            }

            client.reset(new lazy_ostream(naming::id_type
                (gid, naming::id_type::unmanaged))); 
        }
    }

    boost::shared_ptr<lazy_ostream> client;
};

lazy_ostream& cerr()
{
    util::static_<cerr_raii, cerr_raii::tag> cerr_;
    return *cerr_.get().client;  
}

void create_cerr()
{ cerr(); }

}}

#endif

