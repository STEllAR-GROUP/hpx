////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach & Katelyn Kufahl
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/version.hpp>
#include <hpx/exception.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/mpi_environment.hpp>
#include <hpx/util/reinitializable_static.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/agas/stubs/symbol_namespace.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/certificate.hpp>
#include <hpx/components/security/signed_type.hpp>
#endif

#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/vector.hpp>

#include <sstream>

namespace hpx { namespace detail
{
    std::string get_locality_base_name();
}}

namespace hpx { namespace agas
{

//typedef components::detail::heap_factory<
//    lcos::detail::promise<
//        response
//      , response
//    >
//  , components::managed_component<
//        lcos::detail::promise<
//            response
//          , response
//        >
//    >
//> response_heap_type;

// TODO: Make assertions exceptions
void early_parcel_sink(
    parcelset::parcel const& p
    )
{ // {{{
        // decode the action-type in the parcel
        actions::action_type act = p.get_action();

        // early parcels should only be plain actions
        HPX_ASSERT(actions::base_action::plain_action == act->get_action_type());

        // early parcels can't have continuations
        HPX_ASSERT(!p.get_continuation());

        // We should not allow any exceptions to escape the execution of the
        // action as this would bring down the ASIO thread we execute in.
        try {
            act->get_thread_function(0)
                (threads::thread_state_ex(threads::wait_signaled));
        }
        catch (...) {
            std::cerr << hpx::diagnostic_information(boost::current_exception())
                      << std::endl;
            std::abort();
        }
//     }
} // }}}

// This structure is used when a locality registers with node zero
// (first round trip)
struct registration_header
{
    registration_header() {}

    // TODO: pass head address as a GVA
    registration_header(
        parcelset::endpoints_type const& endpoints_
      , boost::uint64_t primary_ns_ptr_
      , boost::uint64_t symbol_ns_ptr_
      , boost::uint32_t cores_needed_
      , boost::uint32_t num_threads_
      , std::string const& hostname_
      , naming::gid_type prefix_ = naming::gid_type()
    ) :
        endpoints(endpoints_)
      , primary_ns_ptr(primary_ns_ptr_)
      , symbol_ns_ptr(symbol_ns_ptr_)
      , cores_needed(cores_needed_)
      , num_threads(num_threads_)
      , hostname(hostname_)
      , prefix(prefix_)
    {}

    parcelset::endpoints_type endpoints;
    boost::uint64_t primary_ns_ptr;
    boost::uint64_t symbol_ns_ptr;
    boost::uint32_t cores_needed;
    boost::uint32_t num_threads;
    std::string hostname;           // hostname of locality
    naming::gid_type prefix;        // suggested prefix (optional)

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & endpoints;
        ar & primary_ns_ptr;
        ar & symbol_ns_ptr;
        ar & cores_needed;
        ar & num_threads;
        ar & hostname;
        ar & prefix;
    }
};

// This structure is used in the response from node zero to the locality which
// is trying to register (first roundtrip).
struct notification_header
{
    notification_header() {}

    notification_header(
        naming::gid_type const& prefix_
      , parcelset::locality const & agas_locality_
      , naming::address const& locality_ns_address_
      , naming::address const& primary_ns_address_
      , naming::address const& component_ns_address_
      , naming::address const& symbol_ns_address_
      , boost::uint32_t num_localities_
      , boost::uint32_t used_cores_
      , parcelset::endpoints_type const & agas_endpoints_
    ) :
        prefix(prefix_)
      , agas_locality(agas_locality_)
      , locality_ns_address(locality_ns_address_)
      , primary_ns_address(primary_ns_address_)
      , component_ns_address(component_ns_address_)
      , symbol_ns_address(symbol_ns_address_)
      , num_localities(num_localities_)
      , used_cores(used_cores_)
      , agas_endpoints(agas_endpoints_)
    {}

    naming::gid_type prefix;
    parcelset::locality agas_locality;
    naming::address locality_ns_address;
    naming::address primary_ns_address;
    naming::address component_ns_address;
    naming::address symbol_ns_address;
    boost::uint32_t num_localities;
    boost::uint32_t used_cores;
    parcelset::endpoints_type agas_endpoints;

#if defined(HPX_HAVE_SECURITY)
    components::security::signed_certificate root_certificate;
#endif

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & prefix;
        ar & agas_locality;
        ar & locality_ns_address;
        ar & primary_ns_address;
        ar & component_ns_address;
        ar & symbol_ns_address;
        ar & num_localities;
        ar & used_cores;
        ar & agas_endpoints;
#if defined(HPX_HAVE_SECURITY)
        ar & root_certificate;
#endif
    }
};

// {{{ early action forwards
void register_worker(registration_header const& header);
void notify_worker(notification_header const& header);
// }}}

// {{{ early action types
typedef actions::action<
    void (*)(registration_header const&)
  , register_worker
> register_worker_action;

typedef actions::action<
    void (*)(notification_header const&)
  , notify_worker
> notify_worker_action;
// }}}

#if defined(HPX_HAVE_SECURITY)
// This structure is used when a locality registers with node zero
// (second roundtrip)
struct registration_header_security
{
    registration_header_security() {}

    registration_header_security(
            naming::gid_type const& prefix_
          , parcelset::endpoints_type const& endpoints_
          , components::security::signed_type<
                components::security::certificate_signing_request> const& csr_)
      : prefix(prefix_)
      , endpoints(endpoints_)
      , csr(csr_)
    {}

    naming::gid_type prefix;
    parcelset::endpoints_type endpoints;

    // CSR for sub-CA of the locality which tries to register
    components::security::signed_type<
        components::security::certificate_signing_request> csr;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & prefix;
        ar & endpoints;
        ar & csr;
    }
};

// This structure is used in the response from node zero to the locality which
// is trying to register.
struct notification_header_security
{
    notification_header_security() {}

    notification_header_security(
            components::security::signed_certificate const& root_subca_certificate_,
            components::security::signed_certificate const& subca_certificate_)
      : root_subca_certificate(root_subca_certificate_)
      , subca_certificate(subca_certificate_)
    {}

    components::security::signed_certificate root_subca_certificate;
    components::security::signed_certificate subca_certificate;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int)
    {
        ar & root_subca_certificate;
        ar & subca_certificate;
    }
};

void register_worker_security(registration_header_security const& header);
void notify_worker_security(notification_header_security const& header);

typedef actions::action<
    void (*)(registration_header_security const&)
  , register_worker_security
> register_worker_security_action;

typedef actions::action<
    void (*)(notification_header_security const&)
  , notify_worker_security
> notify_worker_security_action;
#endif
}}

using hpx::agas::register_worker_action;
using hpx::agas::notify_worker_action;

HPX_ACTION_HAS_CRITICAL_PRIORITY(register_worker_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(notify_worker_action);

HPX_REGISTER_PLAIN_ACTION(register_worker_action,
    register_worker_action, hpx::components::factory_enabled)
HPX_REGISTER_PLAIN_ACTION(notify_worker_action,
    notify_worker_action, hpx::components::factory_enabled)

#if defined(HPX_HAVE_SECURITY)
using hpx::agas::register_worker_security_action;
using hpx::agas::notify_worker_security_action;

HPX_ACTION_HAS_CRITICAL_PRIORITY(register_worker_security_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(notify_worker_security_action);

HPX_REGISTER_PLAIN_ACTION(register_worker_security_action,
    register_worker_security_action, hpx::components::factory_enabled)
HPX_REGISTER_PLAIN_ACTION(notify_worker_security_action,
    notify_worker_security_action, hpx::components::factory_enabled)
#endif

namespace hpx { namespace agas
{

// remote call to AGAS
void register_worker(registration_header const& header)
{
    // This lock acquires the bbb mutex on creation. When it goes out of scope,
    // its dtor calls big_boot_barrier::notify().
    big_boot_barrier::scoped_lock lock(get_big_boot_barrier());

    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    if (HPX_UNLIKELY(agas_client.is_connecting()))
    {
        HPX_THROW_EXCEPTION(
            internal_server_error
          , "agas::register_worker"
          , "a locality in connect mode cannot be an AGAS server.");
    }

    if (HPX_UNLIKELY(!agas_client.is_bootstrap()))
    {
        HPX_THROW_EXCEPTION(
            internal_server_error
          , "agas::register_worker"
          , "registration parcel received by non-bootstrap locality.");
    }

    naming::gid_type prefix = header.prefix;
    if (prefix != naming::invalid_gid && naming::get_locality_id_from_gid(prefix) == 0)
    {
        HPX_THROW_EXCEPTION(internal_server_error
            , "agas::register_worker"
            , boost::str(
                boost::format("worker node (%s) can't suggest locality_id zero, "
                "this is reserved for the console") %
                    header.endpoints));
        return;
    }

    if (!agas_client.register_locality(header.endpoints, prefix, header.num_threads))
    {
        HPX_THROW_EXCEPTION(internal_server_error
            , "agas::register_worker"
            , boost::str(
                boost::format("attempt to register locality %s more than once") %
                    header.endpoints));
        return;
    }

    naming::gid_type primary_ns_gid(
        stubs::primary_namespace::get_service_instance(prefix));
    naming::address primary_ns_address(prefix
      , components::get_component_type<agas::server::primary_namespace>()
      , header.primary_ns_ptr);
    agas_client.bind_local(primary_ns_gid, primary_ns_address);

    naming::gid_type symbol_ns_gid(
        stubs::symbol_namespace::get_service_instance(prefix));
    naming::address symbol_ns_address(prefix
      , components::get_component_type<agas::server::symbol_namespace>()
      , header.symbol_ns_ptr);
    agas_client.bind_local(symbol_ns_gid, symbol_ns_address);

    naming::address locality_addr(hpx::get_locality(),
        server::locality_namespace::get_component_type(),
            agas_client.get_bootstrap_locality_ns_ptr());
    naming::address primary_addr(hpx::get_locality(),
        server::primary_namespace::get_component_type(),
            agas_client.get_bootstrap_primary_ns_ptr());
    naming::address component_addr(hpx::get_locality(),
        server::component_namespace::get_component_type(),
            agas_client.get_bootstrap_component_ns_ptr());
    naming::address symbol_addr(hpx::get_locality(),
        server::symbol_namespace::get_component_type(),
            agas_client.get_bootstrap_symbol_ns_ptr());

    // assign cores to the new locality
    boost::uint32_t first_core = rt.assign_cores(header.hostname,
        header.cores_needed);

    big_boot_barrier & bbb = get_big_boot_barrier();

    notification_header hdr (prefix, bbb.here(), locality_addr, primary_addr
      , component_addr, symbol_addr, rt.get_config().get_num_localities()
      , first_core, bbb.get_endpoints());

#if defined(HPX_HAVE_SECURITY)
    // wait for the root certificate to be available
    bool got_root_certificate = false;
    for (std::size_t i = 0; i != HPX_MAX_NETWORK_RETRIES; ++i)
    {
        error_code ec(lightweight);
        hdr.root_certificate = rt.get_root_certificate(ec);
        if (!ec)
        {
            got_root_certificate = true;
            break;
        }
        boost::this_thread::sleep(boost::get_system_time() +
            boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
    }

    if (!got_root_certificate)
    {
        HPX_THROW_EXCEPTION(internal_server_error
          , "agas::register_console"
          , "could not obtain root certificate");
        return;
    }
#endif

    parcelset::locality dest;
    parcelset::locality here = bbb.here();
    BOOST_FOREACH(parcelset::locality const & loc, header.endpoints)
    {
        if(loc.get_type() == here.get_type())
        {
            dest = loc;
            break;
        }
    }

    actions::base_action* p =
        new actions::transfer_action<notify_worker_action>(hdr);


    // TODO: Handle cases where localities try to connect to AGAS while it's
    // shutting down.
    if (agas_client.get_status() != starting)
    {
        // We can just send the parcel now, the connecting locality isn't a part
        // of startup synchronization.
        get_big_boot_barrier().apply_late(
            0
          , naming::get_locality_id_from_gid(prefix)
          , dest, p);
    }

    else
    {
        // AGAS is starting up; this locality is participating in startup
        // synchronization.
#if defined(HPX_HAVE_SECURITY)
        // send response directly to initiate second round trip
        get_big_boot_barrier().apply(
            0
          , naming::get_locality_id_from_gid(prefix)
          , dest, p);
#else
        // delay the final response until the runtime system is up and running
        util::function_nonser<void()>* thunk = new util::function_nonser<void()>(
            boost::bind(
                &big_boot_barrier::apply
              , boost::ref(get_big_boot_barrier())
              , 0
              , naming::get_locality_id_from_gid(prefix)
              , dest
              , p));
        get_big_boot_barrier().add_thunk(thunk);
#endif
    }
}

// AGAS callback to client (first roundtrip response)
void notify_worker(notification_header const& header)
{
    // This lock acquires the bbb mutex on creation. When it goes out of scope,
    // it's dtor calls big_boot_barrier::notify().
    big_boot_barrier::scoped_lock lock(get_big_boot_barrier());

    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    if (HPX_UNLIKELY(agas_client.get_status() != starting))
    {
        std::ostringstream strm;
        strm << "locality " << rt.here() << " has launched early";
        HPX_THROW_EXCEPTION(internal_server_error,
            "agas::notify_worker",
            strm.str());
    }

    util::runtime_configuration& cfg = rt.get_config();

    // set our prefix
    agas_client.set_local_locality(header.prefix);
    cfg.parse("assigned locality",
        boost::str(boost::format("hpx.locality!=%1%")
                  % naming::get_locality_id_from_gid(header.prefix)));

    // store the full addresses of the agas servers in our local service
    agas_client.locality_ns_addr_ = header.locality_ns_address;
    agas_client.primary_ns_addr_ = header.primary_ns_address;
    agas_client.component_ns_addr_ = header.component_ns_address;
    agas_client.symbol_ns_addr_ = header.symbol_ns_address;

    naming::gid_type const& here = hpx::get_locality();

    // register runtime support component
    naming::gid_type runtime_support_gid(header.prefix.get_msb()
      , rt.get_runtime_support_lva());
    naming::address const runtime_support_address(here
      , components::get_component_type<components::server::runtime_support>()
      , rt.get_runtime_support_lva());
    agas_client.bind_local(runtime_support_gid, runtime_support_address);

    runtime_support_gid.set_lsb(boost::uint64_t(0));
    agas_client.bind_local(runtime_support_gid, runtime_support_address);

    naming::gid_type const memory_gid(header.prefix.get_msb()
      , rt.get_memory_lva());
    naming::address const memory_address(here
      , components::get_component_type<components::server::memory>()
      , rt.get_memory_lva());
    agas_client.bind_local(memory_gid, memory_address);

    // register local primary namespace component
    naming::gid_type const primary_gid =
        stubs::primary_namespace::get_service_instance(
            agas_client.get_local_locality());
    naming::address const primary_addr(here
      , server::primary_namespace::get_component_type(),
        agas_client.get_hosted_primary_ns_ptr());
    agas_client.bind_local(primary_gid, primary_addr);

    // register local symbol namespace component
    naming::gid_type const symbol_gid =
        stubs::symbol_namespace::get_service_instance(
            agas_client.get_local_locality());
    naming::address const symbol_addr(here
      , server::symbol_namespace::get_component_type(),
        agas_client.get_hosted_symbol_ns_ptr());
    agas_client.bind_local(symbol_gid, symbol_addr);

    // Assign the initial parcel gid range to the parcelport. Note that we can't
    // get the parcelport through the parcelhandler because it isn't up yet.
    naming::gid_type parcel_lower, parcel_upper;
    agas_client.get_id_range(1000, parcel_lower, parcel_upper);

    rt.get_id_pool().set_range(parcel_lower, parcel_upper);

    //// assign the initial gid range to the unique id range allocator that our
    //// response heap is using
    //response_heap_type::get_heap().set_range(heap_lower, heap_upper);

    //// allocate our first heap
    //response_heap_type::block_type* p = response_heap_type::alloc_heap();

    //// set the base gid that we bound to this heap
    //p->set_gid(heap_lower);

    //// push the heap onto the OSHL
    //response_heap_type::get_heap().add_heap(p);

    //// bind range of GIDs to head addresses
    //agas_client.bind_range(
    //    heap_lower
    //  , response_heap_type::block_type::heap_step
    //  , p->get_address()
    //  , response_heap_type::block_type::heap_size);

    // store number of initial localities
    cfg.set_num_localities(header.num_localities);

    // store number of used cores by other localities
    cfg.set_first_used_core(header.used_cores);
    rt.assign_cores();

    rt.get_parcel_handler().set_resolved_localities(naming::get_gid_from_locality_id(0), header.agas_endpoints);

#if defined(HPX_HAVE_SECURITY)
    // initialize certificate store
    rt.store_root_certificate(header.root_certificate);

    // initiate second round trip to root
    registration_header_security hdr(
        header.prefix
      , rt.here()
      , rt.get_certificate_signing_request());

    get_big_boot_barrier().apply(
        naming::get_locality_id_from_gid(header.prefix)
      , 0
      , header.agas_locality
      , new actions::transfer_action<register_worker_security_action>(hdr));
#endif
}
// }}}

#if defined(HPX_HAVE_SECURITY)
// remote call to AGAS (initiate second roundtrip)
void register_worker_security(registration_header_security const& header)
{
    // This lock acquires the bbb mutex on creation. When it goes out of scope,
    // it's dtor calls big_boot_barrier::notify().
    big_boot_barrier::scoped_lock lock(get_big_boot_barrier());

    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    if (HPX_UNLIKELY(agas_client.is_connecting()))
    {
        HPX_THROW_EXCEPTION(
            internal_server_error
          , "agas::register_worker"
          , "runtime_mode_connect can't find running application.");
    }

    if (HPX_UNLIKELY(!agas_client.is_bootstrap()))
    {
        HPX_THROW_EXCEPTION(
            internal_server_error
          , "agas::register_worker"
          , "registration parcel received by non-bootstrap locality.");
    }

    notification_header_security hdr(
        rt.get_certificate()
      , rt.sign_certificate_signing_request(header.csr));

    actions::base_action* p =
        new actions::transfer_action<notify_worker_security_action>(hdr);


    parcelset::locality dest;
    parcelset::locality here = bbb.here();
    BOOST_FOREACH(parcelset::locality const & loc, header.endpoints)
    {
        if(loc.get_type() == here.get_type())
        {
            dest = loc;
            break;
        }
    }

    // TODO: Handle cases where localities try to connect to AGAS while it's
    // shutting down.
    if (agas_client.get_status() != starting)
    {
        // We can just send the parcel now, the connecting locality isn't a part
        // of startup synchronization.
        get_big_boot_barrier().apply(
            0
          , naming::get_locality_id_from_gid(header.prefix)
          , dest, p);
    }

    else
    {
        // AGAS is starting up; this locality is participating in startup
        // synchronization.
        util::function_nonser<void()>* thunk = new util::function_nonser<void()>(
            boost::bind(
                &big_boot_barrier::apply
              , boost::ref(get_big_boot_barrier())
              , 0
              , naming::get_locality_id_from_gid(header.prefix)
              , dest
              , p));
        get_big_boot_barrier().add_thunk(thunk);
    }
}

// AGAS callback to client
void notify_worker_security(notification_header_security const& header)
{
    // This lock acquires the bbb mutex on creation. When it goes out of scope,
    // it's dtor calls big_boot_barrier::notify().
    big_boot_barrier::scoped_lock lock(get_big_boot_barrier());

    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    if (HPX_UNLIKELY(agas_client.get_status() != starting))
    {
        std::ostringstream strm;
        strm << "locality " << rt.here() << " has launched early";
        HPX_THROW_EXCEPTION(internal_server_error,
            "agas::notify_worker",
            strm.str());
    }

    // finish initializing the certificate store
    rt.store_subordinate_certificate(
        header.root_subca_certificate
      , header.subca_certificate);
}
// }}}
#endif

///////////////////////////////////////////////////////////////////////////////
void big_boot_barrier::spin()
{
    boost::mutex::scoped_lock lock(mtx);
    while (connected)
        cond.wait(lock);
}

inline std::size_t get_number_of_bootstrap_connections(
    util::runtime_configuration const& ini)
{
    service_mode service_type = ini.get_agas_service_mode();
    std::size_t result = 1;

    if (service_mode_bootstrap == service_type)
    {
        std::size_t num_localities =
            static_cast<std::size_t>(ini.get_num_localities());
        result = num_localities ? num_localities-1 : 0;
    }

#if defined(HPX_HAVE_SECURITY)
    result *= 2;        // we have to do 2 roundtrips
#endif

    return result;
}

big_boot_barrier::big_boot_barrier(
    parcelset::parcelport& pp_
  , parcelset::endpoints_type const& endpoints_
  , util::runtime_configuration const& ini_
):
    pp(pp_)
  , endpoints(endpoints_)
  , service_type(ini_.get_agas_service_mode())
  , bootstrap_agas(pp_.agas_locality(ini_))
  , cond()
  , mtx()
  , connected(get_number_of_bootstrap_connections(ini_))
  , thunks(32)
{
    pp_.register_event_handler(&early_parcel_sink);
}

void big_boot_barrier::apply(
    boost::uint32_t source_locality_id
  , boost::uint32_t target_locality_id
  , parcelset::locality const & dest
  , actions::base_action* act
) { // {{{
    naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
    parcelset::parcel p(naming::get_id_from_locality_id(target_locality_id), addr, act);
    if (!p.get_parcel_id())
        p.set_parcel_id(parcelset::parcel::generate_unique_id(source_locality_id));
    pp.send_early_parcel(dest, p);
} // }}}

void big_boot_barrier::apply_late(
    boost::uint32_t source_locality_id
  , boost::uint32_t target_locality_id
  , parcelset::locality const & dest
  , actions::base_action* act
) { // {{{
    naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
    parcelset::parcel p(naming::get_id_from_locality_id(target_locality_id), addr, act);
    if (!p.get_parcel_id())
        p.set_parcel_id(parcelset::parcel::generate_unique_id(source_locality_id));
    get_runtime().get_parcel_handler().put_parcel(p);
} // }}}

void big_boot_barrier::wait_bootstrap()
{ // {{{
    HPX_ASSERT(service_mode_bootstrap == service_type);

    // the root just waits until all localities have connected
    spin();
} // }}}

namespace detail
{
    boost::uint32_t get_number_of_pus_in_cores(boost::uint32_t num_cores)
    {
        threads::topology& top = threads::create_topology();

        boost::uint32_t num_pus = 0;
        for (boost::uint32_t i = 0; i != num_cores; ++i)
        {
            boost::uint32_t num_pus_core = static_cast<boost::uint32_t>(
                top.get_number_of_core_pus(std::size_t(i)));
            if (num_pus_core == ~boost::uint32_t(0))
                return num_cores;       // assume one pu per core

            num_pus += num_pus_core;
        }

        return num_pus;
    }
}

void big_boot_barrier::wait_hosted(std::string const& locality_name,
    void* primary_ns_server, void* symbol_ns_server)
{ // {{{
    HPX_ASSERT(service_mode_bootstrap != service_type);

    // any worker sends a request for registration and waits
    HPX_ASSERT(0 != primary_ns_server);
    HPX_ASSERT(0 != symbol_ns_server);

    runtime& rt = get_runtime();

    // get the number of cores we need for our locality. This respects the
    // affinity description. Cores that are partially used are counted as well
    boost::uint32_t cores_needed = rt.assign_cores();
    boost::uint32_t num_threads =
        boost::uint32_t(rt.get_config().get_os_thread_count());

    naming::gid_type suggested_prefix;

#if defined(HPX_PARCELPORT_MPI)
    // if MPI parcelport is enabled we use the MPI rank as the suggested locality_id
    if (util::mpi_environment::rank() != -1)
    {
        suggested_prefix = naming::get_gid_from_locality_id(
            util::mpi_environment::rank());
    }
#endif

    // contact the bootstrap AGAS node
    registration_header hdr(
          rt.endpoints()
        , reinterpret_cast<boost::uint64_t>(primary_ns_server)
        , reinterpret_cast<boost::uint64_t>(symbol_ns_server)
        , cores_needed
        , num_threads
        , locality_name
        , suggested_prefix);

    apply(
        naming::invalid_locality_id
        , 0
        , bootstrap_agas
        , new actions::transfer_action<register_worker_action>(hdr));

    // wait for registration to be complete
    spin();
} // }}}

void big_boot_barrier::notify()
{
    boost::mutex::scoped_lock lk(mtx, boost::adopt_lock);
    --connected;
    cond.notify_all();
}

// This is triggered in runtime_impl::start, after the early action handler
// has been replaced by the parcelhandler. We have to delay the notifications
// until this point so that the AGAS locality can come up.
void big_boot_barrier::trigger()
{
    if (service_mode_bootstrap == service_type)
    {
        util::function_nonser<void()>* p;

        while (thunks.pop(p))
            (*p)();
    }
}

///////////////////////////////////////////////////////////////////////////////
struct bbb_tag;

void create_big_boot_barrier(
    parcelset::parcelport& pp_
  , parcelset::endpoints_type const& endpoints_
  , util::runtime_configuration const& ini_
) {
    util::reinitializable_static<boost::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "create_big_boot_barrier",
            "create_big_boot_barrier was called more than once");
    }
    bbb.get().reset(new big_boot_barrier(pp_, endpoints_, ini_));
}

void destroy_big_boot_barrier()
{
    util::reinitializable_static<boost::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (!bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "destroy_big_boot_barrier",
            "big_boot_barrier has not been created yet");
    }
    bbb.get().reset();
}

big_boot_barrier& get_big_boot_barrier()
{
    util::reinitializable_static<boost::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (!bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "get_big_boot_barrier",
            "big_boot_barrier has not been created yet");
    }
    return *(bbb.get());
}

}}

