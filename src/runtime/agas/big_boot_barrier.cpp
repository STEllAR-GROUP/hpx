////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach & Katelyn Kufahl
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/detail/hosted_component_namespace.hpp>
#include <hpx/runtime/agas/detail/hosted_locality_namespace.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/put_parcel.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_id_factory.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/yield_k.hpp>
#include <hpx/util/format.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/reinitializable_static.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/safe_lexical_cast.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace detail
{
    std::string get_locality_base_name();
}}

namespace hpx { namespace agas { namespace detail
{
    void register_unassigned_typenames()
    {
        // supposed to be run on locality 0 before
        // before locality communication
        hpx::serialization::detail::id_registry& serialization_registry =
            hpx::serialization::detail::id_registry::instance();

        serialization_registry.fill_missing_typenames();

        hpx::actions::detail::action_registry& action_registry =
            hpx::actions::detail::action_registry::instance();
        action_registry.fill_missing_typenames();
    }

    ///////////////////////////////////////////////////////////////////////////
    struct unassigned_typename_sequence
    {
        unassigned_typename_sequence() {}

        unassigned_typename_sequence(bool /*dummy*/)
          : serialization_typenames(hpx::serialization::detail::id_registry::
                instance().get_unassigned_typenames())
          , action_typenames(hpx::actions::detail::action_registry::
                instance().get_unassigned_typenames())
        {}

        void save(hpx::serialization::output_archive& ar, unsigned) const
        {
            // part running on worker node
            HPX_ASSERT(!action_typenames.empty());
            ar << serialization_typenames;
            ar << action_typenames;
        }

        void load(hpx::serialization::input_archive& ar, unsigned)
        {
            // part running on locality 0
            ar >> serialization_typenames;
            ar >> action_typenames;
        }
        HPX_SERIALIZATION_SPLIT_MEMBER();

        std::vector<std::string> serialization_typenames;
        std::vector<std::string> action_typenames;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct assigned_id_sequence
    {
        assigned_id_sequence() {}

        assigned_id_sequence(unassigned_typename_sequence const& typenames)
        {
            register_ids_on_main_loc(typenames);
        }

        void save(hpx::serialization::output_archive& ar, unsigned) const
        {
            HPX_ASSERT(!action_ids.empty());
            ar << serialization_ids;      // part running on locality 0
            ar << action_ids;
        }

        void load(hpx::serialization::input_archive& ar, unsigned)
        {
            ar >> serialization_ids;      // part running on worker node
            ar >> action_ids;
        }
        HPX_SERIALIZATION_SPLIT_MEMBER();

    private:
        void register_ids_on_main_loc(
            unassigned_typename_sequence const& unassigned_ids)
        {
            {
                hpx::serialization::detail::id_registry& registry =
                    hpx::serialization::detail::id_registry::instance();
                std::uint32_t max_id = registry.get_max_registered_id();

                for (const std::string& s : unassigned_ids.serialization_typenames)
                {
                    std::uint32_t id = registry.try_get_id(s);
                    if (id == hpx::serialization::detail::id_registry::invalid_id)
                    {
                        // this id is not registered yet
                        id = ++max_id;
                        registry.register_typename(s, id);
                    }
                    serialization_ids.push_back(id);
                }
            }
            {
                hpx::actions::detail::action_registry& registry =
                    hpx::actions::detail::action_registry::instance();
                std::uint32_t max_id = registry.max_id_;

                for (const std::string& s : unassigned_ids.action_typenames)
                {
                    std::uint32_t id = registry.try_get_id(s);
                    if (id == hpx::actions::detail::action_registry::invalid_id)
                    {
                        // this id is not registered yet
                        id = ++max_id;
                        registry.register_typename(s, id);
                    }
                    action_ids.push_back(id);
                }
            }
        }

    public:
        void register_ids_on_worker_loc() const
        {
            {
                hpx::serialization::detail::id_registry& registry =
                    hpx::serialization::detail::id_registry::instance();

                // Yes, we look up the unassigned typenames twice, but this allows
                // to avoid using globals and protects from race conditions during
                // de-serialization.
                std::vector<std::string> typenames =
                    registry.get_unassigned_typenames();

                // we should have received as many ids as we have unassigned names
                HPX_ASSERT(typenames.size() == serialization_ids.size());

                for (std::size_t k = 0; k < serialization_ids.size(); ++k)
                {
                    registry.register_typename(typenames[k], serialization_ids[k]);
                }

                // fill in holes which might have been caused by initialization
                // order problems
                registry.fill_missing_typenames();
            }
            {
                hpx::actions::detail::action_registry& registry =
                    hpx::actions::detail::action_registry::instance();

                // Yes, we look up the unassigned typenames twice, but this allows
                // to avoid using globals and protects from race conditions during
                // de-serialization.
                std::vector<std::string> typenames =
                    registry.get_unassigned_typenames();

                // we should have received as many ids as we have unassigned names
                HPX_ASSERT(typenames.size() == action_ids.size());

                for (std::size_t k = 0; k < action_ids.size(); ++k)
                {
                    registry.register_typename(typenames[k], action_ids[k]);
                }

                // fill in holes which might have been caused by initialization
                // order problems
                registry.fill_missing_typenames();
            }
        }

        std::vector<std::uint32_t> serialization_ids;
        std::vector<std::uint32_t> action_ids;
    };
}}} // namespace hpx::agas::detail

namespace hpx { namespace agas
{

    template <typename Action, typename... Args>
    void big_boot_barrier::apply(
        std::uint32_t source_locality_id
      , std::uint32_t target_locality_id
      , parcelset::locality dest
      , Action act
      , Args &&... args)
    { // {{{
        HPX_ASSERT(pp);
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));
        parcelset::parcel p(
            parcelset::detail::create_parcel::call(std::false_type(),
                naming::get_gid_from_locality_id(target_locality_id),
                std::move(addr), act, std::forward<Args>(args)...));
#if defined(HPX_HAVE_PARCEL_PROFILING)
        if (!p.parcel_id())
        {
            p.parcel_id() = parcelset::parcel::generate_unique_id(source_locality_id);
        }
#endif

        parcelset::detail::parcel_await_apply(std::move(p),
            parcelset::write_handler_type(), 0,
            [this, dest](parcelset::parcel&& p, parcelset::write_handler_type&&)
            {
                pp->send_early_parcel(dest, std::move(p));
            });
    } // }}}

    template <typename Action, typename... Args>
    void big_boot_barrier::apply_late(
        std::uint32_t source_locality_id
      , std::uint32_t target_locality_id
      , parcelset::locality const & dest
      , Action act
      , Args &&... args)
    { // {{{
        naming::address addr(naming::get_gid_from_locality_id(target_locality_id));

        parcelset::put_parcel(
            naming::id_type(
                naming::get_gid_from_locality_id(target_locality_id),
                naming::id_type::unmanaged),
            std::move(addr), act, std::forward<Args>(args)...);
    } // }}}

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

// This structure is used when a locality registers with node zero
// (first round trip)
struct registration_header
{
    registration_header()
      : primary_ns_ptr(0)
      , symbol_ns_ptr(0)
      , cores_needed(0)
      , num_threads(0)
    {}

    // TODO: pass head address as a GVA
    registration_header(
          parcelset::endpoints_type const& endpoints_
        , std::uint64_t primary_ns_ptr_
        , std::uint64_t symbol_ns_ptr_
        , std::uint32_t cores_needed_
        , std::uint32_t num_threads_
        , std::string const& hostname_
        , detail::unassigned_typename_sequence const& typenames_
        , naming::gid_type prefix_ = naming::gid_type())
      : endpoints(endpoints_)
      , primary_ns_ptr(primary_ns_ptr_)
      , symbol_ns_ptr(symbol_ns_ptr_)
      , cores_needed(cores_needed_)
      , num_threads(num_threads_)
      , hostname(hostname_)
      , typenames(typenames_)
      , prefix(prefix_)
    {}

    parcelset::endpoints_type endpoints;
    std::uint64_t primary_ns_ptr;
    std::uint64_t symbol_ns_ptr;
    std::uint32_t cores_needed;
    std::uint32_t num_threads;
    std::string hostname;           // hostname of locality
    detail::unassigned_typename_sequence typenames;
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
        ar & typenames;
        ar & prefix;
    }
};

// This structure is used in the response from node zero to the locality which
// is trying to register (first roundtrip).
struct notification_header
{
    notification_header()
      : num_localities(0)
      , used_cores(0)
    {}

    notification_header(
          naming::gid_type const& prefix_
        , parcelset::locality const & agas_locality_
        , naming::address const& locality_ns_address_
        , naming::address const& primary_ns_address_
        , naming::address const& component_ns_address_
        , naming::address const& symbol_ns_address_
        , std::uint32_t num_localities_
        , std::uint32_t used_cores_
        , parcelset::endpoints_type const & agas_endpoints_
        , detail::assigned_id_sequence const & ids_)
      : prefix(prefix_)
      , agas_locality(agas_locality_)
      , locality_ns_address(locality_ns_address_)
      , primary_ns_address(primary_ns_address_)
      , component_ns_address(component_ns_address_)
      , symbol_ns_address(symbol_ns_address_)
      , num_localities(num_localities_)
      , used_cores(used_cores_)
      , agas_endpoints(agas_endpoints_)
      , ids(ids_)
    {}

    naming::gid_type prefix;
    parcelset::locality agas_locality;
    naming::address locality_ns_address;
    naming::address primary_ns_address;
    naming::address component_ns_address;
    naming::address symbol_ns_address;
    std::uint32_t num_localities;
    std::uint32_t used_cores;
    parcelset::endpoints_type agas_endpoints;
    detail::assigned_id_sequence ids;
    std::vector<parcelset::endpoints_type> endpoints;

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
        ar & ids;
        ar & endpoints;
    }
};

// {{{ early action forwards
void register_worker(registration_header const& header);
void notify_worker(notification_header const& header);
// }}}

// {{{ early action types
typedef actions::direct_action<
    void (*)(registration_header const&)
  , register_worker
> register_worker_action;

typedef actions::direct_action<
    void (*)(notification_header const&)
  , notify_worker
> notify_worker_action;
// }}}

}}

using hpx::agas::register_worker_action;
using hpx::agas::notify_worker_action;

HPX_ACTION_HAS_CRITICAL_PRIORITY(register_worker_action);
HPX_ACTION_HAS_CRITICAL_PRIORITY(notify_worker_action);

HPX_REGISTER_ACTION_ID(register_worker_action,
    register_worker_action,
    hpx::actions::register_worker_action_id)
HPX_REGISTER_ACTION_ID(notify_worker_action,
    notify_worker_action,
    hpx::actions::notify_worker_action_id)

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
            , hpx::util::format(
                "worker node ({}) can't suggest locality_id zero, "
                "this is reserved for the console",
                header.endpoints));
        return;
    }

    if (!agas_client.register_locality(header.endpoints, prefix, header.num_threads))
    {
        HPX_THROW_EXCEPTION(internal_server_error
            , "agas::register_worker"
            , hpx::util::format(
                "attempt to register locality {} more than once",
                header.endpoints));
        return;
    }

//     naming::gid_type primary_ns_gid(
//         primary_namespace::get_service_instance(prefix));
//     naming::address primary_ns_address(prefix
//       , components::get_component_type<agas::server::primary_namespace>()
//       , header.primary_ns_ptr);
//     agas_client.bind_local(primary_ns_gid, primary_ns_address);

//     naming::gid_type symbol_ns_gid(
//         symbol_namespace::get_service_instance(prefix));
//     naming::address symbol_ns_address(prefix
//       , components::get_component_type<agas::server::symbol_namespace>()
//       , header.symbol_ns_ptr);
//     agas_client.bind_local(symbol_ns_gid, symbol_ns_address);

    naming::address locality_addr(hpx::get_locality(),
        hpx::components::component_agas_locality_namespace,
            agas_client.locality_ns_->ptr());
    naming::address primary_addr(hpx::get_locality(),
        hpx::components::component_agas_primary_namespace,
            agas_client.primary_ns_.ptr());
    naming::address component_addr(hpx::get_locality(),
        hpx::components::component_agas_component_namespace,
            agas_client.component_ns_->ptr());
    naming::address symbol_addr(hpx::get_locality(),
        hpx::components::component_agas_symbol_namespace,
            agas_client.symbol_ns_.ptr());

    // assign cores to the new locality
    std::uint32_t first_core = rt.assign_cores(header.hostname,
        header.cores_needed);

    big_boot_barrier & bbb = get_big_boot_barrier();

    // register all ids
    detail::assigned_id_sequence assigned_ids(header.typenames);

    notification_header hdr (prefix, bbb.here(), locality_addr, primary_addr
      , component_addr, symbol_addr, rt.get_config().get_num_localities()
      , first_core, bbb.get_endpoints(), assigned_ids);

    parcelset::locality dest;
    parcelset::locality here = bbb.here();
    for (parcelset::endpoints_type::value_type const & loc : header.endpoints)
    {
        if(loc.second.type() == here.type())
        {
            dest = loc.second;
            break;
        }
    }

    // collect endpoints from all registering localities
    bbb.add_locality_endpoints(naming::get_locality_id_from_gid(prefix),
        header.endpoints);

    // TODO: Handle cases where localities try to connect to AGAS while it's
    // shutting down.
    if (agas_client.get_status() != state_starting)
    {
        // We can just send the parcel now, the connecting locality isn't a part
        // of startup synchronization.
        get_big_boot_barrier().apply_late(
            0
          , naming::get_locality_id_from_gid(prefix)
          , dest
          , notify_worker_action()
          , std::move(hdr));
    }

    else
    {
        // AGAS is starting up; this locality is participating in startup
        // synchronization.

        // delay the final response until the runtime system is up and running
        util::unique_function_nonser<void()>* thunk =
            new util::unique_function_nonser<void()>(
                util::bind(
                    util::one_shot(&big_boot_barrier::apply_notification)
                  , std::ref(get_big_boot_barrier())
                  , 0
                  , naming::get_locality_id_from_gid(prefix)
                  , dest
                  , std::move(hdr)));
        get_big_boot_barrier().add_thunk(thunk);
    }
}

// AGAS callback to client (first round trip response)
void notify_worker(notification_header const& header)
{
    // This lock acquires the bbb mutex on creation. When it goes out of scope,
    // it's dtor calls big_boot_barrier::notify().
    big_boot_barrier::scoped_lock lock(get_big_boot_barrier());

    // register all ids with this locality
    header.ids.register_ids_on_worker_loc();

    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    if (HPX_UNLIKELY(agas_client.get_status() != state_starting))
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
    agas_client.register_console(header.agas_endpoints);
    cfg.parse("assigned locality",
        hpx::util::format("hpx.locality!={1}",
            naming::get_locality_id_from_gid(header.prefix)));

    // store the full addresses of the agas servers in our local service
    agas_client.component_ns_.reset(
        new detail::hosted_component_namespace(header.component_ns_address));
    agas_client.locality_ns_.reset(
        new detail::hosted_locality_namespace(header.locality_ns_address));
    naming::gid_type const& here = hpx::get_locality();

    // register runtime support component
    naming::gid_type runtime_support_gid(header.prefix.get_msb()
      , rt.get_runtime_support_lva());
    naming::address const runtime_support_address(here
      , components::get_component_type<components::server::runtime_support>()
      , rt.get_runtime_support_lva());
    agas_client.bind_local(runtime_support_gid, runtime_support_address);

    runtime_support_gid.set_lsb(std::uint64_t(0));
    agas_client.bind_local(runtime_support_gid, runtime_support_address);

    naming::gid_type const memory_gid(header.prefix.get_msb()
      , rt.get_memory_lva());
    naming::address const memory_address(here
      , components::get_component_type<components::server::memory>()
      , rt.get_memory_lva());
    agas_client.bind_local(memory_gid, memory_address);

    // Assign the initial parcel gid range to the parcelport. Note that we can't
    // get the parcelport through the parcelhandler because it isn't up yet.
    naming::gid_type parcel_lower, parcel_upper;
    agas_client.get_id_range(1000, parcel_lower, parcel_upper);

    rt.get_id_pool().set_range(parcel_lower, parcel_upper);

    // store number of initial localities
    cfg.set_num_localities(header.num_localities);

    // store number of used cores by other localities
    cfg.set_first_used_core(header.used_cores);
    rt.assign_cores();

    // pre-cache all known locality endpoints in local AGAS
    agas_client.pre_cache_endpoints(header.endpoints);
}
// }}}

void big_boot_barrier::apply_notification(
    std::uint32_t source_locality_id
    , std::uint32_t target_locality_id
    , parcelset::locality const& dest
    , notification_header && hdr)
{
    hdr.endpoints = localities;
    apply(source_locality_id, target_locality_id, dest,
        notify_worker_action(), std::move(hdr));
}

void big_boot_barrier::add_locality_endpoints(std::uint32_t locality_id,
    parcelset::endpoints_type const& endpoints)
{
    if (localities.size() < locality_id + 1)
        localities.resize(locality_id + 1);

    localities[locality_id] = endpoints;
}

///////////////////////////////////////////////////////////////////////////////
void big_boot_barrier::spin()
{
    std::unique_lock<compat::mutex> lock(mtx);
    while (connected)
        cond.wait(lock);

    // pre-cache all known locality endpoints in local AGAS on locality 0 as well
    if (service_mode_bootstrap == service_type)
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();
        agas_client.pre_cache_endpoints(localities);
    }
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

    return result;
}

big_boot_barrier::big_boot_barrier(
    parcelset::parcelport *pp_
  , parcelset::endpoints_type const& endpoints_
  , util::runtime_configuration const& ini_
):
    pp(pp_)
  , endpoints(endpoints_)
  , service_type(ini_.get_agas_service_mode())
  , bootstrap_agas(pp_ ? pp_->agas_locality(ini_) : parcelset::locality())
  , cond()
  , mtx()
  , connected(get_number_of_bootstrap_connections(ini_))
  , thunks(32)
{
    // register all not registered typenames
    if (service_type == service_mode_bootstrap)
    {
        detail::register_unassigned_typenames();
        // store endpoints of root locality for later
        add_locality_endpoints(0, get_endpoints());
    }
}

void big_boot_barrier::wait_bootstrap()
{ // {{{
    HPX_ASSERT(service_mode_bootstrap == service_type);

    // the root just waits until all localities have connected
    spin();
} // }}}

namespace detail
{
    std::uint32_t get_number_of_pus_in_cores(std::uint32_t num_cores)
    {
        threads::topology& top = threads::create_topology();

        std::uint32_t num_pus = 0;
        for (std::uint32_t i = 0; i != num_cores; ++i)
        {
            std::uint32_t num_pus_core = static_cast<std::uint32_t>(
                top.get_number_of_core_pus(std::size_t(i)));
            if (num_pus_core == ~std::uint32_t(0))
                return num_cores;       // assume one pu per core

            num_pus += num_pus_core;
        }

        return num_pus;
    }
}

void big_boot_barrier::wait_hosted(
    std::string const& locality_name,
    naming::address::address_type primary_ns_server,
    naming::address::address_type symbol_ns_server)
{ // {{{
    HPX_ASSERT(service_mode_bootstrap != service_type);

    // any worker sends a request for registration and waits
    HPX_ASSERT(0 != primary_ns_server);
    HPX_ASSERT(0 != symbol_ns_server);

    runtime& rt = get_runtime();

    // get the number of cores we need for our locality. This respects the
    // affinity description. Cores that are partially used are counted as well
    std::uint32_t cores_needed = rt.assign_cores();
    std::uint32_t num_threads =
        std::uint32_t(rt.get_config().get_os_thread_count());

    naming::gid_type suggested_prefix;

    std::string locality_str = rt.get_config().get_entry("hpx.locality", "-1");
    if(locality_str != "-1")
    {
        suggested_prefix = naming::get_gid_from_locality_id(
            util::safe_lexical_cast<std::uint32_t>(locality_str, -1));
    }

    // pre-load all unassigned ids
    detail::unassigned_typename_sequence unassigned(true);

    // contact the bootstrap AGAS node
    registration_header hdr(
          rt.endpoints()
        , primary_ns_server
        , symbol_ns_server
        , cores_needed
        , num_threads
        , locality_name
        , unassigned
        , suggested_prefix);

    apply(
          static_cast<std::uint32_t>(std::random_device{}()) // random first parcel id
        , 0
        , bootstrap_agas
        , register_worker_action()
        , std::move(hdr));

    // wait for registration to be complete
    spin();
} // }}}

void big_boot_barrier::notify()
{
    runtime& rt = get_runtime();
    naming::resolver_client& agas_client = rt.get_agas_client();

    bool notify = false;
    {
        std::lock_guard<compat::mutex> lk(mtx, std::adopt_lock);
        if (agas_client.get_status() == state_starting)
        {
            --connected;
            if (connected == 0)
                notify = true;
        }
    }
    if (notify)
        cond.notify_all();
}

// This is triggered in runtime_impl::start, after the early action handler
// has been replaced by the parcelhandler. We have to delay the notifications
// until this point so that the AGAS locality can come up.
void big_boot_barrier::trigger()
{
    if (service_mode_bootstrap == service_type)
    {
        util::unique_function_nonser<void()>* p;

        while (thunks.pop(p))
        {
            try
            {
                (*p)();
            }
            catch(...)
            {
                delete p;
                throw;
            }
            delete p;
        }
    }
}

void big_boot_barrier::add_thunk(util::unique_function_nonser<void()>* f)
{
    std::size_t k = 0;
    while (!thunks.push(f))
    {
        // Wait until successfully pushed ...
        hpx::util::detail::yield_k(
            k, "hpx::agas::big_boot_barrier::add_thunk");
        ++k;
    }
}

///////////////////////////////////////////////////////////////////////////////
struct bbb_tag;

void create_big_boot_barrier(
    parcelset::parcelport *pp_
  , parcelset::endpoints_type const& endpoints_
  , util::runtime_configuration const& ini_
) {
    util::reinitializable_static<std::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
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
    util::reinitializable_static<std::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
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
    util::reinitializable_static<std::shared_ptr<big_boot_barrier>, bbb_tag> bbb;
    if (!bbb.get())
    {
        HPX_THROW_EXCEPTION(internal_server_error,
            "get_big_boot_barrier",
            "big_boot_barrier has not been created yet");
    }
    return *(bbb.get());
}

}}

