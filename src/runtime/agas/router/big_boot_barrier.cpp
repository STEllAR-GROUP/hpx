////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if HPX_AGAS_VERSION > 0x10

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/uintptr_t.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/agas/router/big_boot_barrier.hpp>

#include <boost/thread.hpp>
#include <boost/assert.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

namespace hpx { namespace agas
{

typedef lcos::eager_future<
    naming::resolver_client::primary_namespace_server_type::bind_locality_action,
    naming::resolver_client::response_type
> allocate_response_future_type;

typedef lcos::eager_future<
    naming::resolver_client::primary_namespace_server_type::bind_gid_action,
    naming::resolver_client::response_type
> bind_response_future_type;

typedef components::heap_factory<
    lcos::detail::future_value<
        naming::resolver_client::response_type
      , naming::resolver_client::response_type
      , 1
    >
  , components::managed_component<
        lcos::detail::future_value<
            naming::resolver_client::response_type
          , naming::resolver_client::response_type
          , 1
        >
    >
> response_heap_type; 

void early_parcel_sink(
    parcelset::parcelport&
  , boost::shared_ptr<std::vector<char> > const& parcel_data
) { // {{{
    parcelset::parcel p;

    typedef util::container_device<std::vector<char> > io_device_type;
    boost::iostreams::stream<io_device_type> io (*parcel_data.get());

    // De-serialize the parcel data
    #if HPX_USE_PORTABLE_ARCHIVES != 0
        util::portable_binary_iarchive archive(io);
    #else
        boost::archive::binary_iarchive archive(io);
    #endif
    
    archive >> p;

    // decode the local virtual address of the parcel
    naming::address addr = p.get_destination_addr();

    // decode the action-type in the parcel
    actions::action_type act = p.get_action();

    // early parcels should only be plain actions
    BOOST_ASSERT
        (actions::base_action::plain_action == act->get_action_type());

    // early parcels can't have continuations 
    BOOST_ASSERT(!p.get_continuation());

    act->get_thread_function(0)
        (threads::thread_state_ex(threads::wait_signaled));
} // }}}

void early_write_handler(
    boost::system::error_code const& e 
  , std::size_t size
) {
/*
    hpx::util::osstream strm;
    strm << e.message() << " (read " 
         << size << " bytes)";
    HPX_THROW_EXCEPTION(network_error, 
        "agas::early_write_handler", 
        hpx::util::osstream_get_string(strm));
*/
}

struct notification_header
{
    notification_header() {}

    notification_header(
        naming::gid_type const& prefix_
      , naming::address const& response_pool_address_
      , hpx::uintptr_t response_heap_ptr_
      , naming::gid_type const& response_lower_gid_
      , naming::gid_type const& response_upper_gid_
      , naming::address const& primary_ns_address_
      , naming::address const& component_ns_address_
      , naming::address const& symbol_ns_address_
    ) :
        prefix(prefix_)
      , response_pool_address(response_pool_address_)
      , response_heap_ptr(response_heap_ptr_)
      , response_lower_gid(response_lower_gid_)
      , response_upper_gid(response_upper_gid_)
      , primary_ns_address(primary_ns_address_)
      , component_ns_address(component_ns_address_)
      , symbol_ns_address(symbol_ns_address_)
    {}

    naming::gid_type prefix;
    naming::address response_pool_address;
    hpx::uintptr_t response_heap_ptr;
    naming::gid_type response_lower_gid;
    naming::gid_type response_upper_gid;
    naming::address primary_ns_address;
    naming::address component_ns_address;
    naming::address symbol_ns_address;

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int) 
    {
        ar & prefix;
        ar & response_pool_address;
        ar & response_heap_ptr;
        ar & response_lower_gid;
        ar & response_upper_gid;
        ar & primary_ns_address;
        ar & component_ns_address;
        ar & symbol_ns_address;
    }
};

// {{{ early action forwards
void register_console(
    boost::uint64_t count
  , naming::address const& baseaddr
  , hpx::uintptr_t offset
  , hpx::uintptr_t heap_ptr
);

void notify_console(notification_header const& header);

void register_worker(
    boost::uint64_t count
  , naming::address const& baseaddr
  , hpx::uintptr_t offset
  , hpx::uintptr_t heap_ptr
);

void notify_worker(notification_header const& header);
// }}}

// {{{ early action types
typedef actions::plain_action4<
    boost::uint64_t, naming::address const&, hpx::uintptr_t, hpx::uintptr_t
  , register_console
> register_console_action;

typedef actions::plain_action1<
    notification_header const&, notify_console
> notify_console_action;

typedef actions::plain_action4<
    boost::uint64_t, naming::address const&, hpx::uintptr_t, hpx::uintptr_t
  , register_worker
> register_worker_action;

typedef actions::plain_action1<
    notification_header const&, notify_worker
> notify_worker_action;
// }}}

// {{{ early action definitions
// remote call to AGAS
// TODO: merge with register_worker which is all but identical
void register_console(
    boost::uint64_t count
  , naming::address const& baseaddr
  , hpx::uintptr_t offset
  , hpx::uintptr_t heap_ptr
) {
    naming::resolver_client& agas_client = get_runtime().get_agas_client();

    if (HPX_UNLIKELY(agas_client.state() != router_state_launching))
    {
        hpx::util::osstream strm;
        strm << "AGAS server has already launched, can't register "
             << baseaddr.locality_; 
        HPX_THROW_EXCEPTION(internal_server_error, 
            "agas::register_console", 
            hpx::util::osstream_get_string(strm));
    }

    naming::gid_type prefix, lower, upper;

    agas_client.get_prefix(baseaddr.locality_, prefix, true, false); 
    agas_client.get_id_range(baseaddr.locality_, count, lower, upper);
    agas_client.bind_range(lower, count, baseaddr, offset); 
    agas_client.registerid("/locality(console)", prefix);

    /*
    if (HPX_UNLIKELY((prefix + 1) != lower))
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "agas::register_console",
            "bad initial GID range allocation");
    }
    */

    typedef naming::resolver_client::primary_namespace_server_type
        primary_namespace_server_type;
    typedef naming::resolver_client::component_namespace_server_type
        component_namespace_server_type;
    typedef naming::resolver_client::symbol_namespace_server_type
        symbol_namespace_server_type;

    naming::address primary_addr(get_runtime().here(),
        primary_namespace_server_type::get_component_type(),
            static_cast<void*>(&agas_client.bootstrap->primary_ns_server));
    naming::address component_addr(get_runtime().here(),
        component_namespace_server_type::get_component_type(), 
            static_cast<void*>(&agas_client.bootstrap->component_ns_server));
    naming::address symbol_addr(get_runtime().here(),
        symbol_namespace_server_type::get_component_type(),
            static_cast<void*>(&agas_client.bootstrap->symbol_ns_server));

    actions::base_action* p =
        new notify_console_action(
            notification_header(
                prefix, baseaddr, heap_ptr, lower, upper,
                primary_addr, component_addr, symbol_addr));

    get_big_boot_barrier().apply(naming::address(baseaddr.locality_), p);

    get_big_boot_barrier().notify();
}

// TODO: merge with notify_worker which is all but identical
// TODO: callback must finishing installing new heap
// TODO: callback must set up future pool 
// AGAS callback to client
void notify_console(notification_header const& header)
{
    naming::resolver_client& agas_client = get_runtime().get_agas_client();

    if (HPX_UNLIKELY(agas_client.state() != router_state_launching))
    {
        hpx::util::osstream strm;
        strm << "locality "
             << get_runtime().here() 
             << " has launched early"; 
        HPX_THROW_EXCEPTION(internal_server_error, 
            "agas::notify_console", 
            hpx::util::osstream_get_string(strm));
    }

    // set our prefix
    agas_client.local_prefix(header.prefix);

    // store the full addresses of the agas servers in our local router
    agas_client.hosted->primary_ns_addr_ = header.primary_ns_address;
    agas_client.hosted->component_ns_addr_ = header.component_ns_address;
    agas_client.hosted->symbol_ns_addr_ = header.symbol_ns_address;

    // assign the initial gid range to the unique id range allocator that our
    // response heap is using
    response_heap_type::get_heap().set_range(header.response_lower_gid
                                           , header.response_upper_gid); 

    // finish setting up the first heap. 
    response_heap_type::block_type* p
        = reinterpret_cast<response_heap_type::block_type*>
            (header.response_heap_ptr);

    // set the base gid that we bound to this heap
    p->set_gid(header.response_lower_gid); 

    // push the heap onto the OSHL
    response_heap_type::get_heap().add_heap(p); 

    // set up the future pools
    naming::resolver_client::allocate_response_pool_type&
        allocate_pool = agas_client.get_allocate_response_pool();
    naming::resolver_client::bind_response_pool_type&
        bind_pool = agas_client.get_bind_response_pool();

    util::runtime_configuration const& ini_ = get_runtime().get_config();

    const std::size_t allocate_size =
        ini_.get_agas_allocate_response_pool_size();
    const std::size_t bind_size =
        ini_.get_agas_bind_response_pool_size();

    for (std::size_t i = 0; i < allocate_size; ++i)
        allocate_pool.enqueue(new allocate_response_future_type);     

    for (std::size_t i = 0; i < bind_size; ++i)
        bind_pool.enqueue(new bind_response_future_type);     

    get_big_boot_barrier().notify();
}

// remote call to AGAS
void register_worker(
    boost::uint64_t count
  , naming::address const& baseaddr
  , hpx::uintptr_t offset
  , hpx::uintptr_t heap_ptr
) {
    naming::resolver_client& agas_client = get_runtime().get_agas_client();

    if (HPX_UNLIKELY(agas_client.state() != router_state_launching))
    {
        hpx::util::osstream strm;
        strm << "AGAS server has already launched, can't register "
             << baseaddr.locality_; 
        HPX_THROW_EXCEPTION(internal_server_error, 
            "agas::register_worker", 
            hpx::util::osstream_get_string(strm));
    }

    naming::gid_type prefix, lower, upper;

    agas_client.get_prefix(baseaddr.locality_, prefix, true, false); 
    agas_client.get_id_range(baseaddr.locality_, count, lower, upper);
    agas_client.bind_range(lower, count, baseaddr, offset); 

    /* 
    if (HPX_UNLIKELY((prefix + 1) != lower))
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "agas::register_worker",
            "bad initial GID range allocation");
    }
    */ 

    typedef naming::resolver_client::primary_namespace_server_type
        primary_namespace_server_type;
    typedef naming::resolver_client::component_namespace_server_type
        component_namespace_server_type;
    typedef naming::resolver_client::symbol_namespace_server_type
        symbol_namespace_server_type;

    naming::address primary_addr(get_runtime().here(),
        primary_namespace_server_type::get_component_type(),
            static_cast<void*>(&agas_client.bootstrap->primary_ns_server));
    naming::address component_addr(get_runtime().here(),
        component_namespace_server_type::get_component_type(), 
            static_cast<void*>(&agas_client.bootstrap->component_ns_server));
    naming::address symbol_addr(get_runtime().here(),
        symbol_namespace_server_type::get_component_type(),
            static_cast<void*>(&agas_client.bootstrap->symbol_ns_server));

    actions::base_action* p =
        new notify_console_action(
            notification_header(
                prefix, baseaddr, heap_ptr, lower, upper,
                primary_addr, component_addr, symbol_addr));

    get_big_boot_barrier().apply(naming::address(baseaddr.locality_), p);

    get_big_boot_barrier().notify();
}

// TODO: callback must finish installing new heap
// TODO: callback must set up future pool 
// AGAS callback to client
void notify_worker(notification_header const& header)
{
    naming::resolver_client& agas_client = get_runtime().get_agas_client();

    if (HPX_UNLIKELY(agas_client.state() != router_state_launching))
    {
        hpx::util::osstream strm;
        strm << "locality "
             << get_runtime().here() 
             << " has launched early"; 
        HPX_THROW_EXCEPTION(internal_server_error, 
            "agas::notify_worker", 
            hpx::util::osstream_get_string(strm));
    }

    // set our prefix
    agas_client.local_prefix(header.prefix);

    // store the full addresses of the agas servers in our local router
    agas_client.hosted->primary_ns_addr_ = header.primary_ns_address;
    agas_client.hosted->component_ns_addr_ = header.component_ns_address;
    agas_client.hosted->symbol_ns_addr_ = header.symbol_ns_address;

    // assign the initial gid range to the unique id range allocator that our
    // response heap is using
    response_heap_type::get_heap().set_range(header.response_lower_gid
                                           , header.response_upper_gid); 

    // finish setting up the first heap. 
    response_heap_type::block_type* p
        = reinterpret_cast<response_heap_type::block_type*>
            (header.response_heap_ptr);

    // set the base gid that we bound to this heap
    p->set_gid(header.response_lower_gid); 

    // push the heap onto the OSHL
    response_heap_type::get_heap().add_heap(p); 

    // set up the future pools
    naming::resolver_client::allocate_response_pool_type&
        allocate_pool = agas_client.get_allocate_response_pool();
    naming::resolver_client::bind_response_pool_type&
        bind_pool = agas_client.get_bind_response_pool();

    util::runtime_configuration const& ini_ = get_runtime().get_config();

    const std::size_t allocate_size =
        ini_.get_agas_allocate_response_pool_size();
    const std::size_t bind_size =
        ini_.get_agas_bind_response_pool_size();

    for (std::size_t i = 0; i < allocate_size; ++i)
        allocate_pool.enqueue(new allocate_response_future_type);     

    for (std::size_t i = 0; i < bind_size; ++i)
        bind_pool.enqueue(new bind_response_future_type);     

    get_big_boot_barrier().notify();
}
// }}}

}}

using hpx::agas::register_console_action;
using hpx::agas::notify_console_action;
using hpx::agas::register_worker_action;
using hpx::agas::notify_worker_action;

HPX_REGISTER_PLAIN_ACTION(register_console_action);
HPX_REGISTER_PLAIN_ACTION(notify_console_action);
HPX_REGISTER_PLAIN_ACTION(register_worker_action);
HPX_REGISTER_PLAIN_ACTION(notify_worker_action);
  
namespace hpx { namespace agas
{

void big_boot_barrier::spin()
{
    boost::unique_lock<boost::mutex> lock(mtx);
    while (connected)
        cond.wait(lock);
}

big_boot_barrier::big_boot_barrier(
    parcelset::parcelport& pp_ 
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
):
    pp(pp_)
  , connection_cache_(pp_.get_connection_cache())
  , io_service_pool_(pp_.get_io_service_pool())
  , router_type(ini_.get_agas_router_mode())
  , runtime_type(runtime_type_)
  , bootstrap_agas(ini_.get_agas_locality())
  , cond()
  , mtx()
  , connected( (router_mode_bootstrap == router_type)
             ? ( ini_.get_num_localities()
               ? (ini_.get_num_localities() - 1)
               : 0)
             : 1) 
{
    pp_.register_event_handler(boost::bind(&early_parcel_sink, _1, _2));
}

void big_boot_barrier::apply(
    naming::address const& addr
  , actions::base_action* act 
) { // {{{
    parcelset::parcel p(addr, act);

    parcelset::parcelport_connection_ptr client_connection
        (connection_cache_.get(addr.locality_));

    if (!client_connection)
    {
        // The parcel gets serialized inside the connection constructor, no 
        // need to keep the original parcel alive after this call returned.
        client_connection.reset(new parcelset::parcelport_connection
            (io_service_pool_.get_io_service(), addr.locality_,
                connection_cache_)); 
        client_connection->set_parcel(p);

        // Connect to the target locality, retry if needed
        boost::system::error_code error = boost::asio::error::try_again;

        for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            try 
            { // {{{
                naming::locality::iterator_type
                    it = addr.locality_.connect_begin
                        (io_service_pool_.get_io_service()), 
                    end = addr.locality_.connect_end();

                for (; it != end; ++it)
                {
                    client_connection->socket().close();
                    client_connection->socket().connect(*it, error);

                    if (!error) 
                        break;
                }

                if (!error) 
                    break;

                // wait for a really short amount of time (usually 100 ms)
                boost::this_thread::sleep(boost::get_system_time() + 
                    boost::posix_time::milliseconds
                        (HPX_NETWORK_RETRIES_SLEEP));
            } // }}}

            catch (boost::system::error_code const& e)
            {
                HPX_THROW_EXCEPTION(network_error, 
                    "big_boot_barrier::get_connection", e.message());
            }
        }

        if (error)
        { // {{{ 
            client_connection->socket().close();

            hpx::util::osstream strm;
            strm << error.message() << " (while trying to connect to: " 
                 << addr.locality_ << ")";
            HPX_THROW_EXCEPTION(network_error,
                "big_boot_barrier::get_connection", 
                hpx::util::osstream_get_string(strm));
        } // }}}
    }

    else
        client_connection->set_parcel(p);

    client_connection->async_write(early_write_handler);
} // }}} 

void big_boot_barrier::wait()
{ // {{{
    if (router_mode_bootstrap == router_type)
        spin();

    else
    {
        // allocate our first heap
        response_heap_type::block_type* p = response_heap_type::alloc_heap();

        // hosted, console
        if (runtime_mode_console == runtime_type)
        {
            // We need to contact the bootstrap AGAS node, and then wait
            // for it to signal us. We do this by executing register_console
            // on the bootstrap AGAS node, and sleeping on this node. We'll
            // be woken up by notify_console. 

            apply(bootstrap_agas, new register_console_action
                ((boost::uint64_t) response_heap_type::block_type::heap_step,
                    p->get_address(), (boost::uint64_t)
                        response_heap_type::block_type::heap_size,
                            (hpx::uintptr_t) p)); 
            spin();
        }

        // hosted, worker
        else
        {
            // we need to contact the bootstrap AGAS node, and then wait
            // for it to signal us. 

            apply(bootstrap_agas, new register_worker_action
                ((boost::uint64_t) response_heap_type::block_type::heap_step,
                    p->get_address(), (boost::uint64_t)
                        response_heap_type::block_type::heap_size,
                            (hpx::uintptr_t) p)); 
            spin();
        }
    }  
} // }}}

void big_boot_barrier::notify()
{
    boost::mutex::scoped_lock lk(mtx);
    --connected;
    cond.notify_all();
}

struct bbb_tag;

void create_big_boot_barrier(
    parcelset::parcelport& pp_ 
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
) {
    util::static_<boost::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error, 
            "create_big_boot_barrier",
            "create_big_boot_barrier was called more than once");
    }
    bbb.get().reset(new big_boot_barrier(pp_, ini_, runtime_type_));
}

big_boot_barrier& get_big_boot_barrier()
{
    util::static_<boost::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (!bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error, 
            "get_big_boot_barrier",
            "big_boot_barrier has not been created yet");
    }
    return *(bbb.get());
}

}}

#endif
