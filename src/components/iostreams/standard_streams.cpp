////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/lcos/local/reinitializable_static.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>
#include <hpx/components/iostreams/standard_streams.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::actions::plain_action0<hpx::iostreams::create_cout>
    create_cout_action;
typedef hpx::actions::plain_action0<hpx::iostreams::create_cerr>
    create_cerr_action;

HPX_REGISTER_PLAIN_ACTION(create_cout_action, create_cout_action,
    hpx::components::factory_enabled)
HPX_REGISTER_PLAIN_ACTION(create_cerr_action, create_cerr_action,
    hpx::components::factory_enabled)

///////////////////////////////////////////////////////////////////////////////
// TODO: Use startup/shutdown functions to properly create hpx::cout and
// hpx::cerr. Also, cleanup on shutdown.

namespace hpx { namespace iostreams
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Tag types to be used for the RAII wrappers below
        struct raii_cout_tag {};
        struct raii_cerr_tag {};

        std::ostream& get_outstream(raii_cout_tag)
        {
            return std::cout;
        }

        std::ostream& get_outstream(raii_cerr_tag)
        {
            return std::cerr;
        }

        char const* const cout_name = "/locality#console/output_stream#cout";
        char const* const cerr_name = "/locality#console/output_stream#cerr";
    }

    ///////////////////////////////////////////////////////////////////////////
    // This is a RAII wrapper managing the output stream objects (or their
    // references) for a particular locality
    template <typename Tag>
    struct stream_raii : boost::noncopyable
    {
        typedef components::managed_component<server::output_stream> ostream_type;

        stream_raii(char const* cout_name)
        {
            LRT_(info) << "stream_raii::stream_raii: creating '"
                       << cout_name << "' stream object";

            naming::resolver_client& agas_client = get_runtime().get_agas_client();
            if (agas_client.is_console())
            {
                naming::id_type cout_id(
                    components::server::create_with_args<ostream_type>(
                        boost::ref(detail::get_outstream(Tag()))),
                    naming::id_type::managed);

                // use async version to force the current thread to be suspended,
                // which avoids deadlocks if executed with one OS-thread
                lcos::future<bool> f = agas::register_name_async(cout_name, cout_id);
                f.get();

                client.reset(new lazy_ostream(cout_id));
            }

            else
            {
                // FIXME: Use an error code here?

                // use async version to force the current thread to be suspended,
                // which avoids deadlocks if executed with one OS-thread
                lcos::future<naming::id_type> f = agas::resolve_name_async(cout_name);
                naming::id_type gid = f.get();

                if (!gid)
                {
                    error_code ec(lightweight);
                    naming::id_type console = agas::get_console_locality(ec);
                    if (HPX_UNLIKELY(ec || !console))
                    {
                        HPX_THROW_EXCEPTION(no_registered_console,
                            "stream_raii::stream_raii", "couldn't contact console");
                    }

                    hpx::async<create_cout_action>(console).get();

                    // Try again
                    f = agas::resolve_name_async(cout_name);
                    gid = f.get(ec);

                    if (HPX_UNLIKELY(ec || !gid))
                    {
                        HPX_THROW_EXCEPTION(service_unavailable,
                            "stream_raii::stream_raii",
                            "couldn't create cout stream on the console locality");
                    }
                }

                client.reset(new lazy_ostream(gid));
            }
        }

        boost::shared_ptr<lazy_ostream> client;
    };

    // return the singleton stream objects
    lazy_ostream& cout()
    {
        typedef lcos::local::reinitializable_static<
            stream_raii<detail::raii_cout_tag>, detail::raii_cout_tag
        > static_type;

        static_type cout_(detail::cout_name);
        return *cout_.get().client;
    }

    lazy_ostream& cerr()
    {
        typedef lcos::local::reinitializable_static<
            stream_raii<detail::raii_cerr_tag>, detail::raii_cerr_tag
        > static_type;

        static_type cerr_(detail::cerr_name);
        return *cerr_.get().client;
    }

    // force the creation of the singleton stream objects
    void create_cout()
    {
        cout();
    }

    void create_cerr()
    {
        cerr();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    cout_wrapper cout = {};
    cerr_wrapper cerr = {};
}

