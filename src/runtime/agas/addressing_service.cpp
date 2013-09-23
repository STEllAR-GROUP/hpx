////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/runtime/agas/big_boot_barrier.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/locality_namespace.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/symbol_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/agas/server/symbol_namespace.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#if !defined(HPX_GCC_VERSION) || (HPX_GCC_VERSION > 40400)
#include <hpx/lcos/broadcast.hpp>
#endif

#include <boost/format.hpp>
#include <boost/icl/closed_interval.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/vector.hpp>

namespace hpx { namespace agas
{

struct addressing_service::bootstrap_data_type
{ // {{{
    bootstrap_data_type()
      : primary_ns_server_()
      , locality_ns_server_(&primary_ns_server_)
      , component_ns_server_()
      , symbol_ns_server_()
    {}

    void register_counter_types()
    {
        server::locality_namespace::register_counter_types();
        server::primary_namespace::register_counter_types();
        server::component_namespace::register_counter_types();
        server::symbol_namespace::register_counter_types();
    }

    void register_server_instance(char const* servicename)
    {
        locality_ns_server_.register_server_instance(servicename);
        primary_ns_server_.register_server_instance(servicename);
        component_ns_server_.register_server_instance(servicename);
        symbol_ns_server_.register_server_instance(servicename);
    }

    void unregister_server_instance(error_code& ec)
    {
        locality_ns_server_.unregister_server_instance(ec);
        if (!ec) primary_ns_server_.unregister_server_instance(ec);
        if (!ec) component_ns_server_.unregister_server_instance(ec);
        if (!ec) symbol_ns_server_.unregister_server_instance(ec);
    }

    server::primary_namespace primary_ns_server_;
    server::locality_namespace locality_ns_server_;
    server::component_namespace component_ns_server_;
    server::symbol_namespace symbol_ns_server_;
}; // }}}

struct addressing_service::hosted_data_type
{ // {{{
    hosted_data_type()
      : primary_ns_server_()
      , symbol_ns_server_()
    {}

    void register_counter_types()
    {
        server::primary_namespace::register_counter_types();
        server::symbol_namespace::register_counter_types();
    }

    void register_server_instance(char const* servicename
      , boost::uint32_t locality_id)
    {
        primary_ns_server_.register_server_instance(servicename, locality_id);
        symbol_ns_server_.register_server_instance(servicename, locality_id);
    }

    void unregister_server_instance(error_code& ec)
    {
        primary_ns_server_.unregister_server_instance(ec);
        if (!ec) symbol_ns_server_.unregister_server_instance(ec);
    }

    locality_namespace locality_ns_;
    component_namespace component_ns_;

    server::primary_namespace primary_ns_server_;
    server::symbol_namespace symbol_ns_server_;
}; // }}}

struct addressing_service::gva_cache_key
{ // {{{ gva_cache_key implementation
  private:
    typedef boost::icl::closed_interval<naming::gid_type, std::less>
        key_type;

    key_type key_;

  public:
    gva_cache_key()
      : key_()
    {}

    explicit gva_cache_key(
        naming::gid_type const& id_
      , boost::uint64_t count_ = 1
        )
      : key_(naming::detail::get_stripped_gid(id_)
           , naming::detail::get_stripped_gid(id_) + (count_ - 1))
    {
        BOOST_ASSERT(count_);
    }

    naming::gid_type get_gid() const
    {
        return boost::icl::lower(key_);
    }

    boost::uint64_t get_count() const
    {
        naming::gid_type const size = boost::icl::length(key_);
        BOOST_ASSERT(size.get_msb() == 0);
        return size.get_lsb();
    }

    friend bool operator<(
        gva_cache_key const& lhs
      , gva_cache_key const& rhs
        )
    {
        return boost::icl::exclusive_less(lhs.key_, rhs.key_);
    }

    friend bool operator==(
        gva_cache_key const& lhs
      , gva_cache_key const& rhs
        )
    {
        // Is lhs in rhs?
        if (1 == lhs.get_count() && 1 != rhs.get_count())
            return boost::icl::contains(rhs.key_, lhs.key_);

        // Is rhs in lhs?
        else if (1 != lhs.get_count() && 1 == rhs.get_count())
            return boost::icl::contains(lhs.key_, lhs.key_);

        // Direct hit
        return lhs.key_ == rhs.key_;
    }
}; // }}}

struct addressing_service::gva_erase_policy
{ // {{{ gva_erase_policy implementation
    gva_erase_policy(
        naming::gid_type const& id
      , boost::uint64_t count
        )
      : entry(id, count)
    {}

    typedef std::pair<
        gva_cache_key, boost::cache::entries::lfu_entry<gva>
    > entry_type;

    bool operator()(
        entry_type const& p
        ) const
    {
        return p.first == entry;
    }

    gva_cache_key entry;
}; // }}}

addressing_service::addressing_service(
    parcelset::parcelport& pp
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
    )
  : gva_cache_(new gva_cache_type)
  , console_cache_(0)
  , max_refcnt_requests_(ini_.get_agas_max_pending_refcnt_requests())
  , refcnt_requests_count_(0)
  , refcnt_requests_(new refcnt_requests_type)
  , service_type(ini_.get_agas_service_mode())
  , runtime_type(runtime_type_)
  , caching_(ini_.get_agas_caching_mode())
  , range_caching_(caching_ ? ini_.get_agas_range_caching_mode() : false)
  , action_priority_(ini_.get_agas_dedicated_server() ?
        threads::thread_priority_normal : threads::thread_priority_critical)
  , here_()         // defer initializing this
  , rts_lva_(0)
  , state_(starting)
  , locality_()
{ // {{{
    create_big_boot_barrier(pp, ini_, runtime_type_);

    if (caching_)
        gva_cache_->reserve(ini_.get_agas_local_cache_size());

    if (service_type == service_mode_bootstrap)
        launch_bootstrap(ini_);

    // now, boot the parcel port
    pp.run(false);
}

void addressing_service::initialize()
{
    if (service_type == service_mode_bootstrap)
    {
        get_big_boot_barrier().wait();
    }
    else
    {
        launch_hosted();
        get_big_boot_barrier().wait(&hosted->primary_ns_server_,
            &hosted->symbol_ns_server_);
    }

    set_status(running);
} // }}}

naming::locality const& addressing_service::get_here() const
{
    if (!here_) {
        runtime* rt = get_runtime_ptr();
        BOOST_ASSERT(rt &&
            rt->get_state() >= runtime::state_initialized &&
            rt->get_state() < runtime::state_stopped);
        here_ = rt->here();
    }
    return here_;
}

void* addressing_service::get_hosted_primary_ns_ptr() const
{
    BOOST_ASSERT(0 != hosted.get());
    return &hosted->primary_ns_server_;
}

void* addressing_service::get_hosted_symbol_ns_ptr() const
{
    BOOST_ASSERT(0 != hosted.get());
    return &hosted->symbol_ns_server_;
}

void* addressing_service::get_bootstrap_locality_ns_ptr() const
{
    BOOST_ASSERT(0 != bootstrap.get());
    return &bootstrap->locality_ns_server_;
}

void* addressing_service::get_bootstrap_primary_ns_ptr() const
{
    BOOST_ASSERT(0 != bootstrap.get());
    return &bootstrap->primary_ns_server_;
}

void* addressing_service::get_bootstrap_component_ns_ptr() const
{
    BOOST_ASSERT(0 != bootstrap.get());
    return &bootstrap->component_ns_server_;
}

void* addressing_service::get_bootstrap_symbol_ns_ptr() const
{
    BOOST_ASSERT(0 != bootstrap.get());
    return &bootstrap->symbol_ns_server_;
}


void addressing_service::launch_bootstrap(
    util::runtime_configuration const& ini_
    )
{ // {{{
    bootstrap = boost::make_shared<bootstrap_data_type>();

    runtime& rt = get_runtime();

    naming::locality const ep = ini_.get_agas_locality();
    naming::gid_type const here =
        naming::get_gid_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX);

    naming::gid_type const locality_gid = bootstrap_locality_namespace_gid();
    gva locality_gva(ep,
        server::locality_namespace::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->locality_ns_server_));
    locality_ns_addr_ = naming::address(ep,
        server::locality_namespace::get_component_type(),
            static_cast<void*>(&bootstrap->locality_ns_server_));

    naming::gid_type const primary_gid = bootstrap_primary_namespace_gid();
    gva primary_gva(ep,
        server::primary_namespace::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->primary_ns_server_));
    primary_ns_addr_ = naming::address(ep,
        server::primary_namespace::get_component_type(),
            static_cast<void*>(&bootstrap->primary_ns_server_));

    naming::gid_type const component_gid = bootstrap_component_namespace_gid();
    gva component_gva(ep,
        server::component_namespace::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->component_ns_server_));
    component_ns_addr_ = naming::address(ep,
        server::component_namespace::get_component_type(),
            static_cast<void*>(&bootstrap->component_ns_server_));

    naming::gid_type const symbol_gid = bootstrap_symbol_namespace_gid();
    gva symbol_gva(ep,
        server::symbol_namespace::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->symbol_ns_server_));
    symbol_ns_addr_ = naming::address(ep,
        server::symbol_namespace::get_component_type(),
            static_cast<void*>(&bootstrap->symbol_ns_server_));

    set_local_locality(here);
    rt.get_config().parse("assigned locality",
        boost::str(boost::format("hpx.locality!=%1%")
                  % naming::get_locality_id_from_gid(here)));

    boost::uint32_t num_threads = boost::lexical_cast<boost::uint32_t>(
        ini_.get_entry("hpx.os_threads", boost::uint32_t(1)));

    request locality_req(locality_ns_allocate, ep, 4, num_threads);
    bootstrap->locality_ns_server_.remote_service(locality_req);

    naming::gid_type runtime_support_gid1(here);
    runtime_support_gid1.set_lsb(rt.get_runtime_support_lva());
    naming::gid_type runtime_support_gid2(here);
    runtime_support_gid2.set_lsb(boost::uint64_t(0));

    gva runtime_support_address(ep
      , components::get_component_type<components::server::runtime_support>()
      , 1U, rt.get_runtime_support_lva());

    request reqs[] =
    {
        request(primary_ns_bind_gid, locality_gid, locality_gva)
      , request(primary_ns_bind_gid, primary_gid, primary_gva)
      , request(primary_ns_bind_gid, component_gid, component_gva)
      , request(primary_ns_bind_gid, symbol_gid, symbol_gva)
//      , request(primary_ns_bind_gid, runtime_support_gid1, runtime_support_address)
//      , request(primary_ns_bind_gid, runtime_support_gid2, runtime_support_address)
    };

    for (std::size_t i = 0; i < (sizeof(reqs) / sizeof(request)); ++i)
        bootstrap->primary_ns_server_.remote_service(reqs[i]);

    register_name("/0/agas/locality#0", here);
    if (is_console())
        register_name("/0/locality#console", here);

    naming::gid_type lower, upper;
    get_id_range(ep, HPX_INITIAL_GID_RANGE, lower, upper);
    rt.get_id_pool().set_range(lower, upper);

//    get_big_boot_barrier().wait();
//    set_status(running);
} // }}}

void addressing_service::launch_hosted()
{ // {{{
    hosted = boost::make_shared<hosted_data_type>();

//    get_big_boot_barrier().wait(&hosted->primary_ns_server_);
//    set_status(running);
} // }}}

void addressing_service::adjust_local_cache_size()
{ // {{{
    // adjust the local AGAS cache size for the number of connected localities
    if (caching_)
    {
        util::runtime_configuration const& cfg = get_runtime().get_config();
        std::size_t local_cache_size = cfg.get_agas_local_cache_size();
        std::size_t local_cache_size_per_thread =
            cfg.get_agas_local_cache_size_per_thread();

        std::size_t cache_size = (std::max)(local_cache_size,
                local_cache_size_per_thread * std::size_t(get_num_overall_threads()));
        if (cache_size > gva_cache_->capacity())
            gva_cache_->reserve(cache_size);

        LAGAS_(info) << (boost::format(
            "addressing_service::adjust_local_cache_size, local_cache_size(%1%), "
            "local_cache_size_per_thread(%2%), cache_size(%3%)")
            % local_cache_size % local_cache_size_per_thread % cache_size);
    }
} // }}}

void addressing_service::set_local_locality(naming::gid_type const& g)
{
    locality_ = g;
    if (is_bootstrap())
        bootstrap->primary_ns_server_.set_local_locality(g);
    else
        hosted->primary_ns_server_.set_local_locality(g);
}

response addressing_service::service(
    request const& req
  , error_code& ec
    )
{ // {{{
    if (req.get_action_code() & primary_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->primary_ns_server_.service(req, ec);
        return hosted->primary_ns_server_.service(req, ec);
    }

    else if (req.get_action_code() & component_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->component_ns_server_.service(req, ec);
        return hosted->component_ns_.service(req, action_priority_, ec);
    }

    else if (req.get_action_code() & symbol_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->symbol_ns_server_.service(req, ec);
        return hosted->symbol_ns_server_.service(req, ec);
    }

    else if (req.get_action_code() & locality_ns_service)
    {
        if (is_bootstrap())
            return bootstrap->locality_ns_server_.service(req, ec);
        return hosted->locality_ns_.service(req, action_priority_, ec);
    }

    HPX_THROWS_IF(ec, bad_action_code
        , "addressing_service::service"
        , "invalid action code encountered in request")
    return response();
} // }}}

std::vector<response> addressing_service::bulk_service(
    std::vector<request> const& req
  , error_code& ec
    )
{ // {{{
    // FIXME: For now, we just send it to the primary namespace, assuming that
    // most requests will end up there anyways. The primary namespace will
    // route the requests to other namespaces (and the other namespaces would
    // also route requests intended for the primary namespace).
    if (is_bootstrap())
        return bootstrap->primary_ns_server_.bulk_service(req, ec);
    return hosted->primary_ns_server_.bulk_service(req, ec);
} // }}}

bool addressing_service::register_locality(
    naming::locality const& ep
  , naming::gid_type& prefix
  , boost::uint32_t num_threads
  , error_code& ec
    )
{ // {{{
    try {
        request req(locality_ns_allocate, ep, 0, num_threads, prefix);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->locality_ns_server_.service(req, ec);
        else
            rep = hosted->locality_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return false;

        prefix = naming::get_gid_from_locality_id(rep.get_locality_id());

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_locality");
        return false;
    }
} // }}}

boost::uint32_t addressing_service::resolve_locality(
    naming::locality const& ep
  , error_code& ec
    )
{ // {{{
    try {
        request req(locality_ns_resolve_locality, ep);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->locality_ns_server_.service(req, ec);
        else
            rep = hosted->locality_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return 0;

        return rep.get_locality_id();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_locality");
        return 0;
    }
} // }}}

// TODO: We need to ensure that the locality isn't unbound while it still holds
// referenced objects.
bool addressing_service::unregister_locality(
    naming::locality const& ep
  , error_code& ec
    )
{ // {{{
    try {
        request req(locality_ns_free, ep);
        response rep;

        if (is_bootstrap())
            bootstrap->unregister_server_instance(ec);
        else
            hosted->unregister_server_instance(ec);

        if (ec)
            return false;

        if (is_bootstrap())
            rep = bootstrap->locality_ns_server_.service(req, ec);
        else
            rep = hosted->locality_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return false;

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unregister_locality");
        return false;
    }
} // }}}

bool addressing_service::get_console_locality(
    naming::gid_type& prefix
  , error_code& ec
    )
{ // {{{
    try {
        if (get_status() != running)
        {
            if (&ec != &throws)
                ec = make_success_code();
            return false;
        }

        if (is_console())
        {
            prefix = get_local_locality();
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }

        {
            mutex_type::scoped_lock lock(console_cache_mtx_);

            if (console_cache_)
            {
                prefix = naming::get_gid_from_locality_id(console_cache_);
                if (&ec != &throws)
                    ec = make_success_code();
                return true;
            }
        }

        std::string key("/0/locality#console");

        request req(symbol_ns_resolve, key);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->symbol_ns_server_.service(req, ec);
        else
            rep = stubs::symbol_namespace::service(key, req, action_priority_, ec);

        if (!ec && (rep.get_gid() != naming::invalid_gid) &&
            (rep.get_status() == success))
        {
            prefix = rep.get_gid();
            boost::uint32_t console = naming::get_locality_id_from_gid(prefix);

            {
                mutex_type::scoped_lock lock(console_cache_mtx_);
                if (!console_cache_) {
                    console_cache_ = console;
                }
                else {
                    BOOST_ASSERT(console_cache_ == console);
                }
            }

            LAS_(debug) <<
                ( boost::format("caching console locality, prefix(%1%)")
                % console);

            return true;
        }

        return false;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_console_locality");
        return false;
    }
} // }}}

bool addressing_service::get_localities(
    std::vector<naming::gid_type>& locality_ids
  , components::component_type type
  , error_code& ec
    )
{ // {{{ get_locality_ids implementation
    try {
        if (type != components::component_invalid)
        {
            request req(component_ns_resolve_id, type);
            response rep;

            if (is_bootstrap())
                rep = bootstrap->component_ns_server_.service(req, ec);
            else
                rep = hosted->component_ns_.service(req, action_priority_, ec);

            if (ec || (success != rep.get_status()))
                return false;

            const std::vector<boost::uint32_t> p = rep.get_localities();

            if (!p.size())
                return false;

            locality_ids.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                locality_ids.push_back(naming::get_gid_from_locality_id(p[i]));

            return true;
        }

        else
        {
            request req(locality_ns_localities);
            response rep;

            if (is_bootstrap())
                rep = bootstrap->locality_ns_server_.service(req, ec);
            else
                rep = hosted->locality_ns_.service(req, action_priority_, ec);

            if (ec || (success != rep.get_status()))
                return false;

            const std::vector<boost::uint32_t> p = rep.get_localities();

            if (!p.size())
                return false;

            locality_ids.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                locality_ids.push_back(naming::get_gid_from_locality_id(p[i]));

            return true;
        }
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_locality_ids");
        return false;
    }
} // }}}

lcos::future<std::vector<naming::locality> >
    addressing_service::get_resolved_localities_async()
{ // {{{ get_resolved_localities_async implementation
    naming::id_type const target = bootstrap_locality_namespace_id();
    request req(locality_ns_resolved_localities);
    return stubs::locality_namespace::service_async<
        std::vector<naming::locality> >(target, req);
} //}}}

///////////////////////////////////////////////////////////////////////////////
boost::uint32_t addressing_service::get_num_localities(
    components::component_type type
  , error_code& ec
    )
{ // {{{ get_num_localities implementation
    try {
        if (type == components::component_invalid)
        {
            request req(locality_ns_num_localities, type);
            response rep;

            if (is_bootstrap())
                rep = bootstrap->locality_ns_server_.service(req, ec);
            else
                rep = hosted->locality_ns_.service(req, action_priority_, ec);

            if (ec || (success != rep.get_status()))
                return boost::uint32_t(-1);

            return rep.get_num_localities();
        }

        request req(component_ns_num_localities, type);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->component_ns_server_.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return boost::uint32_t(-1);

        return rep.get_num_localities();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_localities");
    }
    return boost::uint32_t(-1);
} // }}}

lcos::future<boost::uint32_t> addressing_service::get_num_localities_async(
    components::component_type type
    )
{ // {{{ get_num_localities implementation
    if (type == components::component_invalid)
    {
        naming::id_type const target = bootstrap_locality_namespace_id();
        request req(locality_ns_num_localities, type);
        return stubs::locality_namespace::service_async<boost::uint32_t>(target, req);
    }

    naming::id_type const target = bootstrap_component_namespace_id();
    request req(component_ns_num_localities, type);
    return stubs::component_namespace::service_async<boost::uint32_t>(target, req);
} // }}}

///////////////////////////////////////////////////////////////////////////////
boost::uint32_t addressing_service::get_num_overall_threads(
    error_code& ec
    )
{ // {{{ get_num_overall_threads implementation
    try {
        request req(locality_ns_num_threads);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->locality_ns_server_.service(req, ec);
        else
            rep = hosted->locality_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return boost::uint32_t(0);

        return rep.get_num_overall_threads();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_overall_threads");
    }
    return boost::uint32_t(0);
} // }}}

lcos::future<boost::uint32_t> addressing_service::get_num_overall_threads_async()
{ // {{{
    naming::id_type const target = bootstrap_locality_namespace_id();
    request req(locality_ns_num_threads);
    return stubs::locality_namespace::service_async<boost::uint32_t>(target, req);
} // }}}

std::vector<boost::uint32_t> addressing_service::get_num_threads(
    error_code& ec
    )
{ // {{{ get_num_threads implementation
    try {
        request req(locality_ns_num_threads);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->locality_ns_server_.service(req, ec);
        else
            rep = hosted->locality_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return std::vector<boost::uint32_t>();

        return rep.get_num_threads();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_num_threads");
    }
    return std::vector<boost::uint32_t>();
} // }}}

lcos::future<std::vector<boost::uint32_t> > addressing_service::get_num_threads_async()
{ // {{{
    naming::id_type const target = bootstrap_locality_namespace_id();
    request req(locality_ns_num_threads);
    return stubs::locality_namespace::service_async<
        std::vector<boost::uint32_t> >(target, req);
} // }}}

///////////////////////////////////////////////////////////////////////////////
components::component_type addressing_service::get_component_id(
    std::string const& name
  , error_code& ec
    )
{ /// {{{
    try {
        request req(component_ns_bind_name, name);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->component_ns_server_.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status()))
            return components::component_invalid;

        return rep.get_component_type();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_component_id");
        return components::component_invalid;
    }
} // }}}

void addressing_service::iterate_types(
    iterate_types_function_type const& f
  , error_code& ec
    )
{ // {{{
    try {
        request req(component_ns_iterate_types, f);

        if (is_bootstrap())
            bootstrap->component_ns_server_.service(req, ec);
        else
            hosted->component_ns_.service(req, action_priority_, ec);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_types");
    }
} // }}}

std::string addressing_service::get_component_type_name(
    components::component_type id
  , error_code& ec
    )
{ // {{{
    try {
        request req(component_ns_get_component_type_name, id);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->component_ns_server_.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, action_priority_, ec);

        return rep.get_component_typename();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_types");
    }
    return "<unknown>";
} // }}}

components::component_type addressing_service::register_factory(
    boost::uint32_t prefix
  , std::string const& name
  , error_code& ec
    )
{ // {{{
    try {
        request req(component_ns_bind_prefix, name, prefix);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->component_ns_server_.service(req, ec);
        else
            rep = hosted->component_ns_.service(req, action_priority_, ec);

        if (ec || (success != rep.get_status() && no_success != rep.get_status()))
            return components::component_invalid;

        return rep.get_component_type();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_factory");
        return components::component_invalid;
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::get_id_range(
    boost::uint64_t count
  , naming::gid_type& lower_bound
  , naming::gid_type& upper_bound
  , error_code& ec
    )
{ // {{{ get_id_range implementation
    try {
        // naming::locality() is an obsolete, dummy argument
        request req(primary_ns_allocate, naming::locality(), count, boost::uint32_t(-1));
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server_.service(req, ec);
        else
            rep = hosted->primary_ns_server_.service(req, ec);

        error const s = rep.get_status();

        if (ec || (success != s && repeated_request != s))
            return false;

        lower_bound = rep.get_lower_bound();
        upper_bound = rep.get_upper_bound();

        return success == s;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::get_id_range");
        return false;
    }
} // }}}

bool addressing_service::bind_range(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address const& baseaddr
  , boost::uint64_t offset
  , error_code& ec
    )
{ // {{{ bind_range implementation
    try {
        naming::locality const& ep = baseaddr.locality_;

        // Create a global virtual address from the legacy calling convention
        // parameters.
        gva const g(ep, baseaddr.type_, count, baseaddr.address_, offset);

        request req(primary_ns_bind_gid, lower_id, g);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server_.service(req, ec);
        else
            rep = hosted->primary_ns_server_.service(req, ec);

        error const s = rep.get_status();

        if (ec || (success != s && repeated_request != s))
            return false;

        if (caching_)
        {
            if (range_caching_)
                // Put the range into the cache.
                update_cache_entry(lower_id, g, ec);

            else
            {
                // Only put the first GID in the range into the cache.
                gva const first_g = g.resolve(lower_id, lower_id);
                update_cache_entry(lower_id, first_g, ec);
            }
        }

        if (ec)
            return false;

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::bind_range");
        return false;
    }
} // }}}

bool addressing_service::unbind_range(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ unbind_range implementation
    try {
        request req(primary_ns_unbind_gid, lower_id, count);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server_.service(req, ec);
        else
            rep = hosted->primary_ns_server_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return false;

        // I'm afraid that this will break the first form of paged caching,
        // so it's commented out for now.
        //cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
        //gva_erase_policy ep(lower_id, count);
        //hosted->gva_cache_->erase(ep);

        gva const& gaddr = rep.get_gva();
        addr.locality_ = gaddr.endpoint;
        addr.type_ = gaddr.type;
        addr.address_ = gaddr.lva();

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unbind_range");
        return false;
    }
} // }}}

/// This function will test whether the given address refers to an object
/// living on the locality of the caller. We rely completely on the local AGAS
/// cache and local AGAS instance, assuming that everything which is not in
/// the cache is not local.

bool addressing_service::is_local_address(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{
    // Resolve the address of the GID.

    // NOTE: We do not throw here for a reason; it is perfectly valid for the
    // GID to not be found in the local AGAS instance.
    if (!resolve(id, addr, ec) || ec)
        return false;

    return addr.locality_ == get_here();
}

bool addressing_service::is_local_address_cached(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{
    // Try to resolve the address of the GID.

    // NOTE: We do not throw here for a reason; it is perfectly valid for the
    // GID to not be found in the cache.
    if (!resolve_cached(id, addr, ec) || ec)
        return false;

    return addr.locality_ == get_here();
}

// Return true if at least one address is local.
bool addressing_service::is_local_address(
    naming::gid_type const* gids
  , naming::address* addrs
  , std::size_t size
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    // Try the cache
    if (caching_)
    {
        bool all_resolved = resolve_cached(gids, addrs, size, locals, ec);
        if (ec)
            return false;
        if (all_resolved)
            return locals.any();      // all destinations resolved
    }

    if (!resolve_full(gids, addrs, size, locals, ec) || ec)
        return false;

    return locals.any();
}

bool addressing_service::is_local_lva_encoded_address(
    boost::uint64_t msb
    )
{
    // NOTE: This should still be migration safe.
    return msb == get_local_locality().get_msb();
}

bool addressing_service::resolve_locally_known_addresses(
    naming::gid_type const& id
  , naming::address& addr
    )
{
    // LVA-encoded GIDs (located on this machine)
    boost::uint64_t lsb = id.get_lsb();
    boost::uint64_t msb = naming::detail::strip_credit_from_gid(id.get_msb());

    if (is_local_lva_encoded_address(msb))
    {
        addr.locality_ = get_here();

        // An LSB of 0 references the runtime support component
        if (!rts_lva_)
            rts_lva_ = get_runtime().get_runtime_support_lva();

        if (0 == lsb || lsb == rts_lva_)
        {
            addr.type_ = components::component_runtime_support;
            addr.address_ = rts_lva_;
        }
        else
        {
            addr.type_ = components::component_memory;
            addr.address_ = lsb;
        }

        return true;
    }

    // authoritative AGAS component address resolution
    if (HPX_AGAS_LOCALITY_NS_MSB == msb && HPX_AGAS_LOCALITY_NS_LSB == lsb)
    {
        addr = locality_ns_addr_;
        return true;
    }

    if (HPX_AGAS_PRIMARY_NS_MSB == msb && HPX_AGAS_PRIMARY_NS_LSB == lsb)
    {
        addr = primary_ns_addr_;
        return true;
    }

    if (HPX_AGAS_COMPONENT_NS_MSB == msb && HPX_AGAS_COMPONENT_NS_LSB == lsb)
    {
        addr = component_ns_addr_;
        return true;
    }

    if (HPX_AGAS_SYMBOL_NS_MSB == msb && HPX_AGAS_SYMBOL_NS_LSB == lsb)
    {
        addr = symbol_ns_addr_;
        return true;
    }

    return false;
} // }}}

bool addressing_service::resolve_full(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ resolve implementation
    try {
        // special cases
        if (resolve_locally_known_addresses(id, addr))
            return true;

        request req(primary_ns_resolve_gid, id);
        response rep;

        if (is_bootstrap())
            rep = bootstrap->primary_ns_server_.service(req, ec);
        else
            rep = hosted->primary_ns_server_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return false;

        // Resolve the gva to the real resolved address (which is just a gva
        // with as fully resolved LVA and an offset of zero).
        gva const g = rep.get_gva().resolve(id, rep.get_base_gid());

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva();

        if (caching_)
        {
            if (range_caching_)
                // Put the gva range into the cache.
                update_cache_entry(rep.get_base_gid(), rep.get_gva(), ec);
            else
                // Put the fully resolved gva into the cache.
                update_cache_entry(id, g, ec);
        }

        if (ec)
            return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_full");
        return false;
    }
} // }}}

bool addressing_service::resolve_cached(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ resolve_cached implementation

    // special cases
    if (resolve_locally_known_addresses(id, addr))
        return true;
    if (ec) return false;

    // If caching is disabled, bail
    if (!caching_)
    {
        if (&ec != &throws)
            ec = make_success_code();
        return false;
    }

    // first look up the requested item in the cache
    gva_cache_key k(id);
    gva_cache_key idbase;
    gva_cache_type::entry_type e;

    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

    // Check if the entry is currently in the cache
    if (gva_cache_->get_entry(k, idbase, e))
    {
        const boost::uint64_t id_msb
            = naming::detail::strip_credit_from_gid(id.get_msb());

        if (HPX_UNLIKELY(id_msb != idbase.get_gid().get_msb()))
        {
            HPX_THROWS_IF(ec, internal_server_error
              , "addressing_service::resolve_cached"
              , "bad entry in cache, MSBs of GID base and GID do not match");
            return false;
        }

        gva const& g = e.get();

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva(id, idbase.get_gid());

        lock.unlock();

        if (&ec != &throws)
            ec = make_success_code();

        LAS_(debug) <<
            ( boost::format(
                "cache hit for address %1%, lva %2% (base %3%, lva %4%)")
            % id
            % reinterpret_cast<void*>(addr.address_)
            % idbase.get_gid()
            % reinterpret_cast<void*>(g.lva()));

        return true;
    }

    if (&ec != &throws)
        ec = make_success_code();

    LAS_(debug) << (boost::format("cache miss for address %1%") % id);

    return false;
} // }}}

hpx::future<naming::address> addressing_service::resolve_async(
    naming::gid_type const& gid
    )
{
    // Try the cache.
    if (caching_)
    {
        naming::address addr;
        error_code ec;
        if (resolve_cached(gid, addr, ec))
            return make_ready_future(addr);

        if (ec)
        {
            return make_error_future<naming::address>(
                hpx::detail::access_exception(ec));
        }
    }

    // now try the AGAS service
    return resolve_full_async(gid);
}

naming::address addressing_service::resolve_full_postproc(
    future<response>& f, naming::gid_type const& id
    )
{
    naming::address addr;

    response rep = f.get();
    if (success != rep.get_status())
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_full_postproc",
            "could no resolve global id");
        return addr;
    }

    // Resolve the gva to the real resolved address (which is just a gva
    // with as fully resolved LVA and an offset of zero).
    gva const g = rep.get_gva().resolve(id, rep.get_base_gid());

    addr.locality_ = g.endpoint;
    addr.type_ = g.type;
    addr.address_ = g.lva();

    if (caching_)
    {
        if (range_caching_)
            // Put the gva range into the cache.
            update_cache_entry(rep.get_base_gid(), rep.get_gva());
        else
            // Put the fully resolved gva into the cache.
            update_cache_entry(id, g);
    }

    return addr;
}

hpx::future<naming::address> addressing_service::resolve_full_async(
    naming::gid_type const& gid
    )
{
    // handle special cases
    naming::address addr;
    if (resolve_locally_known_addresses(gid, addr))
        return make_ready_future(addr);

    // ask server
    request req(primary_ns_resolve_gid, gid);
    naming::id_type target(
        stubs::primary_namespace::get_service_instance(gid)
      , naming::id_type::unmanaged);

    using util::placeholders::_1;
    future<response> f = stubs::primary_namespace::service_async<response>(target, req);
    return f.then(util::bind(&addressing_service::resolve_full_postproc, this, _1, gid));
}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::resolve_full(
    naming::gid_type const* gids
  , naming::address* addrs
  , std::size_t count
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    locals.resize(count);

    try {
        std::vector<request> reqs;
        reqs.reserve(count);

        // special cases
        for (std::size_t i = 0; i != count; ++i)
        {
            if (!addrs[i])
            {
                bool is_local = resolve_locally_known_addresses(gids[i], addrs[i]);
                locals.set(i, is_local);
            }
            else
            {
                locals.set(i, true);
            }

            if (!addrs[i] && !locals.test(i))
            {
                reqs.push_back(request(primary_ns_resolve_gid, gids[i]));
            }
        }

        if (reqs.empty()) {
            // all gids have been resolved
            if (&ec != &throws)
                ec = make_success_code();
            return true;
        }

        std::vector<response> reps;
        if (is_bootstrap())
            reps = bootstrap->primary_ns_server_.bulk_service(reqs, ec);
        else
            reps = hosted->primary_ns_server_.bulk_service(reqs, ec);

        if (ec)
            return false;

        std::size_t j = 0;
        for (std::size_t i = 0; i != count; ++i)
        {
            if (addrs[i] || locals.test(i))
                continue;

            BOOST_ASSERT(j < reps.size());
            if (success != reps[j].get_status())
                return false;

            // Resolve the gva to the real resolved address (which is just a gva
            // with as fully resolved LVA and an offset of zero).
            gva const g = reps[j].get_gva().resolve(gids[i], reps[j].get_base_gid());

            naming::address& addr = addrs[i];
            addr.locality_ = g.endpoint;
            addr.type_ = g.type;
            addr.address_ = g.lva();

            if (caching_)
            {
                if (range_caching_) {
                    // Put the gva range into the cache.
                    update_cache_entry(reps[j].get_base_gid(), reps[j].get_gva(), ec);
                }
                else {
                    // Put the fully resolved gva into the cache.
                    update_cache_entry(gids[i], g, ec);
                }
            }

            if (ec)
                return false;

            ++j;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_full");
        return false;
    }
}

bool addressing_service::resolve_cached(
    naming::gid_type const* gids
  , naming::address* addrs
  , std::size_t count
  , boost::dynamic_bitset<>& locals
  , error_code& ec
    )
{
    locals.resize(count);

    std::size_t resolved = 0;
    for (std::size_t i = 0; i != count; ++i)
    {
        if (!addrs[i] && !locals.test(i))
        {
            bool was_resolved = resolve_cached(gids[i], addrs[i], ec);
            if (ec)
                return false;
            if (was_resolved)
                ++resolved;

            if (addrs[i].locality_ == get_here())
                locals.set(i, true);
        }

        else if (addrs[i].locality_ == get_here())
        {
            ++resolved;
            locals.set(i, true);
        }
    }

    return resolved == count;   // returns whether all have been resolved
}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::route(
    parcelset::parcel const& p
  , HPX_STD_FUNCTION<void(boost::system::error_code const&, std::size_t)> const& f
    )
{
    // compose request
    request req(primary_ns_route, p);
    naming::gid_type const* gids = p.get_destinations();

    naming::id_type const target(
        stubs::primary_namespace::get_service_instance(gids[0])
      , naming::id_type::unmanaged);

    typedef server::primary_namespace::service_action action_type;

    // Determine whether the gid is local or remote
    naming::address addr;
    if (is_local_address(target.get_gid(), addr))
    {
        // route through the local AGAS service instance
        applier::detail::apply_l_p<action_type>(addr, action_priority_, req);
        f(boost::system::error_code(), 0);      // invoke callback
        return;
    }

    // apply remotely, route through main AGAS service if the destination is
    // a service instance
    if (!addr)
    {
        if (stubs::primary_namespace::is_service_instance(gids[0]) ||
            stubs::symbol_namespace::is_service_instance(gids[0]))
        {
            // construct wrapper parcel
            parcelset::parcel route_p(bootstrap_primary_namespace_gid()
              , primary_ns_addr_
              , new hpx::actions::transfer_action<action_type>(action_priority_, req));

            // send to the main AGAS instance for routing
            hpx::applier::get_applier().get_parcel_handler().put_parcel(route_p, f);
            return;
        }
    }

    // apply directly as we have the resolved destination address
    applier::detail::apply_r_p_cb<action_type>(addr, target, action_priority_
      , f, req);
}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::incref_apply(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credit
    )
{ // {{{ incref implementation
    if (HPX_UNLIKELY(0 >= credit))
    {
        HPX_THROW_EXCEPTION(bad_parameter
          , "addressing_service::incref_apply"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return;
    }

    request req(primary_ns_change_credit_non_blocking, lower, upper, credit);
    naming::id_type target(
        stubs::primary_namespace::get_service_instance(lower)
      , naming::id_type::unmanaged);

    stubs::primary_namespace::service_non_blocking(target, req, action_priority_);
} // }}}

///////////////////////////////////////////////////////////////////////////////
lcos::future<bool> addressing_service::incref_async(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credit
    )
{ // {{{ incref implementation
    if (HPX_UNLIKELY(0 >= credit))
    {
        HPX_THROW_EXCEPTION(bad_parameter
          , "addressing_service::incref_async"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return lcos::future<bool>();
    }

    request req(primary_ns_change_credit_non_blocking, lower, upper, credit);
    naming::id_type target(
        stubs::primary_namespace::get_service_instance(lower)
      , naming::id_type::unmanaged);

    return stubs::primary_namespace::service_async<bool>(target, req);
} // }}}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::decref(
    naming::gid_type const& lower
  , naming::gid_type const& upper
  , boost::int64_t credit
  , error_code& ec
    )
{ // {{{ decref implementation
    if (HPX_UNLIKELY(0 == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        threads::register_thread_nullary(
            HPX_STD_BIND(&addressing_service::decref, this,
                lower, upper, credit, boost::ref(throws)),
                "addressing_service::decref");
        return;
    }

    if (HPX_UNLIKELY(0 >= credit))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "addressing_service::decref"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return;
    }

    try {
        mutex_type::scoped_lock l(refcnt_requests_mtx_);

        refcnt_requests_->apply(lower, upper
          , util::incrementer<boost::int64_t>(-credit));

        send_refcnt_requests(l, ec);
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::decref");
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::register_name(
    std::string const& name
  , naming::gid_type const& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_bind, name, id);
        response rep;

        if (is_bootstrap() && name.size() >= 2 && name[1] == '0' && name[0] == '/')
            rep = bootstrap->symbol_ns_server_.service(req, ec);
        else
            rep = stubs::symbol_namespace::service(name, req, action_priority_, ec);

        return !ec && (success == rep.get_status());
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::register_name");
        return false;
    }
} // }}}

static void correct_credit_on_failure(future<bool> f, naming::id_type id,
    boost::uint16_t mutable_gid_credit, boost::uint16_t new_gid_credit)
{
    // Return the credit to the GID if the operation failed
    if (f.has_exception() && mutable_gid_credit != 0)
        naming::detail::add_credit_to_gid(id.get_gid(), new_gid_credit);
}

lcos::future<bool> addressing_service::register_name_async(
    std::string const& name
  , naming::id_type const& id
    )
{ // {{{
    // We need to modify the reference count.
    naming::gid_type& mutable_gid = const_cast<naming::id_type&>(id).get_gid();
    naming::gid_type new_gid;

    // FIXME: combine incref with register_name, if needed
    if (naming::detail::get_credit_from_gid(mutable_gid) != 0)
    {
        new_gid = naming::detail::split_credits_for_gid(mutable_gid);

        // Credit exhaustion - we need to get more.
        if (0 == naming::detail::get_credit_from_gid(new_gid))
        {
            BOOST_ASSERT(1 == naming::detail::get_credit_from_gid(mutable_gid));
            naming::get_agas_client().incref(new_gid, 2 * HPX_INITIAL_GLOBALCREDIT);

            naming::detail::add_credit_to_gid(new_gid, HPX_INITIAL_GLOBALCREDIT);
            naming::detail::add_credit_to_gid(mutable_gid, HPX_INITIAL_GLOBALCREDIT);
        }
    }
    else {
        new_gid = mutable_gid;
    }

    request req(symbol_ns_bind, name, new_gid);

    future<bool> f = stubs::symbol_namespace::service_async<bool>(
        name, req, action_priority_);

    using HPX_STD_PLACEHOLDERS::_1;
    f.then(
        HPX_STD_BIND(correct_credit_on_failure, _1, id,
            naming::detail::get_credit_from_gid(mutable_gid),
            naming::detail::get_credit_from_gid(new_gid))
    );
    return f;
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::unregister_name(
    std::string const& name
  , naming::gid_type& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_unbind, name);
        response rep;

        if (is_bootstrap() && name.size() >= 2 && name[1] == '0' && name[0] == '/')
            rep = bootstrap->symbol_ns_server_.service(req, ec);
        else
            rep = stubs::symbol_namespace::service(name, req, action_priority_, ec);

        if (!ec && (success == rep.get_status()))
        {
            id = rep.get_gid();
            return true;
        }

        return false;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::unregister_name");
        return false;
    }
} // }}}

lcos::future<naming::id_type> addressing_service::unregister_name_async(
    std::string const& name
    )
{ // {{{
    request req(symbol_ns_unbind, name);

    return stubs::symbol_namespace::service_async<naming::id_type>(
        name, req, action_priority_);
} // }}}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::resolve_name(
    std::string const& name
  , naming::gid_type& id
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_resolve, name);
        response rep;

        if (is_bootstrap() && name.size() >= 2 && name[1] == '0' && name[0] == '/')
            rep = bootstrap->symbol_ns_server_.service(req, ec);
        else
            rep = stubs::symbol_namespace::service(name, req, action_priority_, ec);

        if (!ec && (success == rep.get_status()))
        {
            id = rep.get_gid();
            return true;
        }

        else
            return false;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_name");
        return false;
    }
} // }}}

lcos::future<naming::id_type> addressing_service::resolve_name_async(
    std::string const& name
    )
{ // {{{
    request req(symbol_ns_resolve, name);

    return stubs::symbol_namespace::service_async<naming::id_type>(
        name, req, action_priority_);
} // }}}

}}

#if !defined(HPX_GCC_VERSION) || (HPX_GCC_VERSION > 40400)

///////////////////////////////////////////////////////////////////////////////
typedef hpx::agas::server::symbol_namespace::service_action
    symbol_namespace_service_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(symbol_namespace_service_action)
HPX_REGISTER_BROADCAST_ACTION(symbol_namespace_service_action)

#endif

namespace hpx { namespace agas
{
/// Invoke the supplied hpx::function for every registered global name
bool addressing_service::iterate_ids(
    iterate_names_function_type const& f
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_iterate_names, f);

#if !defined(HPX_GCC_VERSION) || (HPX_GCC_VERSION > 40400)
        symbol_namespace_service_action act;
        lcos::broadcast(act, hpx::find_all_localities(), req).get(ec);
#else
        BOOST_FOREACH(naming::id_type id, hpx::find_all_localities())
        {
            naming::id_type service_id(
                stubs::symbol_namespace::get_service_instance(id.get_gid(), ec),
                naming::id_type::unmanaged);
            if (ec) return false;

            stubs::symbol_namespace::service(service_id, req, action_priority_, ec);
            if (ec) return false;
        }
#endif

        return !ec;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::iterate_ids");
        return false;
    }
} // }}}

void addressing_service::insert_cache_entry(
    naming::gid_type const& gid
  , gva const& g
  , error_code& ec
    )
{ // {{{
    if (!caching_)
    {
        // If caching is disabled, we silently pretend success.
        return;
    }

    try {
        // The entry in AGAS for a locality's RTS component has a count of 0,
        // so we convert it to 1 here so that the cache doesn't break.
        const boost::uint64_t count = (g.count ? g.count : 1);

        LAS_(debug) <<
            ( boost::format("inserting entry into cache, gid(%1%), count(%2%)")
            % gid % count);

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        const gva_cache_key key(gid, count);

        if (!gva_cache_->insert(key, g))
        {
            // Figure out who we collided with.
            gva_cache_key idbase;
            gva_cache_type::entry_type e;

            if (!gva_cache_->get_entry(key, idbase, e))
                // This is impossible under sane conditions.
                HPX_THROWS_IF(ec, invalid_data
                  , "addressing_service::insert_cache_entry"
                  , "data corruption or lock error occurred in cache");

            LAS_(warning) <<
                ( boost::format(
                    "aborting insert due to key collision in cache, "
                    "new_gid(%1%), new_count(%2%), old_gid(%3%), old_count(%4%)"
                ) % gid % count % idbase.get_gid() % idbase.get_count());
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::insert_cache_entry");
    }
} // }}}

bool check_for_collisions(
    addressing_service::gva_cache_key const& new_key
  , addressing_service::gva_cache_key const& old_key
    )
{
    return (new_key.get_gid() == old_key.get_gid())
        && (new_key.get_count() == old_key.get_count());
}

void addressing_service::update_cache_entry(
    naming::gid_type const& gid
  , gva const& g
  , error_code& ec
    )
{ // {{{
    if (!caching_)
    {
        // If caching is disabled, we silently pretend success.
        return;
    }

    try {
        // The entry in AGAS for a locality's RTS component has a count of 0,
        // so we convert it to 1 here so that the cache doesn't break.
        const boost::uint64_t count = (g.count ? g.count : 1);

        LAS_(debug) <<
            ( boost::format(
                "updating cache entry: gid(%1%), count(%2%)"
            ) % gid % count);

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        const gva_cache_key key(gid, count);

        if (!gva_cache_->update_if(key, g, check_for_collisions))
        {
            // Figure out who we collided with.
            gva_cache_key idbase;
            gva_cache_type::entry_type e;

            if (!gva_cache_->get_entry(key, idbase, e))
            {
                // This is impossible under sane conditions.
                HPX_THROWS_IF(ec, invalid_data
                  , "addressing_service::update_cache_entry"
                  , "data corruption or lock error occurred in cache");
                return;
            }

            LAS_(warning) <<
                ( boost::format(
                    "aborting update due to key collision in cache, "
                    "new_gid(%1%), new_count(%2%), old_gid(%3%), old_count(%4%)"
                ) % gid % count % idbase.get_gid() % idbase.get_count());
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::update_cache_entry");
    }
} // }}}

void addressing_service::clear_cache(
    error_code& ec
    )
{ // {{{
    if (!caching_)
    {
        // If caching is disabled, we silently pretend success.
        return;
    }

    try {
        LAS_(warning) << "clearing cache";

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        gva_cache_->clear();

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::clear_cache");
    }
} // }}}

namespace detail
{
    // get action code from counter type
    namespace_action_code retrieve_action_code(
        std::string const& name
      , error_code& ec
        )
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec) return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0;
             i != num_component_namespace_services;
             ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].code_;
        }

        // locality_ns
        for (std::size_t i = 0;
             i != num_locality_namespace_services;
             ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].code_;
        }

        // primary_ns
        for (std::size_t i = 0;
             i != num_primary_namespace_services;
             ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].code_;
        }

        // symbol_ns
        for (std::size_t i = 0;
             i != num_symbol_namespace_services;
             ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }

    // get service action code from counter type
    namespace_action_code retrieve_action_service_code(
        std::string const& name
      , error_code& ec
        )
    {
        performance_counters::counter_path_elements p;
        performance_counters::get_counter_path_elements(name, p, ec);
        if (ec) return invalid_request;

        if (p.objectname_ != "agas")
        {
            HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
                "unknown performance counter (unrelated to AGAS)");
            return invalid_request;
        }

        // component_ns
        for (std::size_t i = 0;
             i != num_component_namespace_services;
             ++i)
        {
            if (p.countername_ == component_namespace_services[i].name_)
                return component_namespace_services[i].service_code_;
        }

        // locality_ns
        for (std::size_t i = 0;
             i != num_locality_namespace_services;
             ++i)
        {
            if (p.countername_ == locality_namespace_services[i].name_)
                return locality_namespace_services[i].service_code_;
        }

        // primary_ns
        for (std::size_t i = 0;
             i != num_primary_namespace_services;
             ++i)
        {
            if (p.countername_ == primary_namespace_services[i].name_)
                return primary_namespace_services[i].service_code_;
        }

        // symbol_ns
        for (std::size_t i = 0;
             i != num_symbol_namespace_services;
             ++i)
        {
            if (p.countername_ == symbol_namespace_services[i].name_)
                return symbol_namespace_services[i].service_code_;
        }

        HPX_THROWS_IF(ec, bad_parameter, "retrieve_action_service_code",
            "unknown performance counter (unrelated to AGAS)");
        return invalid_request;
    }
}

bool addressing_service::retrieve_statistics_counter(
    std::string const& name
  , naming::gid_type& counter
  , error_code& ec
    )
{
    try {
        // retrieve counter type
        namespace_action_code service_code =
            detail::retrieve_action_service_code(name, ec);
        if (invalid_request == service_code) return false;

        // compose request
        request req(service_code, name);
        response rep;

        if (is_bootstrap() && name.size() >= 2 && name[1] == '0' && name[0] == '/')
            rep = bootstrap->symbol_ns_server_.service(req, ec);
        else
            rep = stubs::symbol_namespace::service(name, req, action_priority_, ec);

        if (!ec && (success == rep.get_status()))
        {
            counter = rep.get_statistics_counter();
            return true;
        }

        return false;
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::query_statistics");
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Helper functions to access the current cache statistics
std::size_t addressing_service::get_cache_hits(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().hits(reset);
}

std::size_t addressing_service::get_cache_misses(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().misses(reset);
}

std::size_t addressing_service::get_cache_evictions(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().evictions(reset);
}

std::size_t addressing_service::get_cache_insertions(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().insertions(reset);
}

/// Install performance counter types exposing properties from the local cache.
void addressing_service::register_counter_types()
{ // {{{
    // install
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_hits(
        boost::bind(&addressing_service::get_cache_hits, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_misses(
        boost::bind(&addressing_service::get_cache_misses, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_evictions(
        boost::bind(&addressing_service::get_cache_evictions, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_insertions(
        boost::bind(&addressing_service::get_cache_insertions, this, ::_1));

    performance_counters::generic_counter_type_data const counter_types[] =
    {
        { "/agas/count/cache-hits", performance_counters::counter_raw,
          "returns the number of cache hits while accessing the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_hits, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache-misses", performance_counters::counter_raw,
          "returns the number of cache misses while accessing the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_misses, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache-evictions", performance_counters::counter_raw,
          "returns the number of cache evictions from the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_evictions, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache-insertions", performance_counters::counter_raw,
          "returns the number of cache insertions into the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insertions, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        }
    };
    performance_counters::install_counter_types(
        counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

    if (is_bootstrap()) {
        // install counters for services
        bootstrap->register_counter_types();

        // always register root server as 'locality#0'
        bootstrap->register_server_instance("locality#0/");
    }
    else {
        // install counters for services
        hosted->register_counter_types();

        boost::uint32_t locality_id =
            naming::get_locality_id_from_gid(get_local_locality());
        std::string str("locality#" + boost::lexical_cast<std::string>(locality_id) + "/");
        hosted->register_server_instance(str.c_str(), locality_id);
    }
} // }}}

void addressing_service::garbage_collect_non_blocking(
    error_code& ec
    )
{
    mutex_type::scoped_lock l(refcnt_requests_mtx_);
    send_refcnt_requests_non_blocking(l, ec);
}

void addressing_service::garbage_collect(
    error_code& ec
    )
{
    mutex_type::scoped_lock l(refcnt_requests_mtx_);
    send_refcnt_requests_sync(l, ec);
}

void addressing_service::send_refcnt_requests(
    addressing_service::mutex_type::scoped_lock& l
  , error_code& ec
    )
{
    if (!l.owns_lock())
    {
        HPX_THROWS_IF(ec, lock_error
          , "addressing_service::send_refcnt_requests"
          , "mutex is not locked");
        return;
    }

    if (max_refcnt_requests_ == ++refcnt_requests_count_)
        send_refcnt_requests_non_blocking(l, ec);

    else if (&ec != &throws)
        ec = make_success_code();
}

void addressing_service::send_refcnt_requests_non_blocking(
    addressing_service::mutex_type::scoped_lock& l
  , error_code& ec
    )
{
    try {
        boost::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

        p.swap(refcnt_requests_);
        refcnt_requests_count_ = 0;

        l.unlock();

        BOOST_FOREACH(refcnt_requests_type::const_reference e, *p)
        {
            naming::gid_type lower(boost::icl::lower(e.key()));
            request const req(
                primary_ns_change_credit_non_blocking
              , lower
              , boost::icl::upper(e.key())
              , e.data());

            naming::id_type target(
                stubs::primary_namespace::get_service_instance(lower)
                , naming::id_type::unmanaged);

            stubs::primary_namespace::service_non_blocking(
                target, req, action_priority_);
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::send_refcnt_requests");
    }
}

void addressing_service::send_refcnt_requests_sync(
    addressing_service::mutex_type::scoped_lock& l
  , error_code& ec
    )
{
    boost::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

    p.swap(refcnt_requests_);
    refcnt_requests_count_ = 0;

    l.unlock();

    std::vector<hpx::future<response> > lazy_results;
    BOOST_FOREACH(refcnt_requests_type::const_reference e, *p)
    {
        naming::gid_type lower(boost::icl::lower(e.key()));
        request const req(primary_ns_change_credit_sync
                        , lower
                        , boost::icl::upper(e.key())
                        , e.data());

        naming::id_type target(
            stubs::primary_namespace::get_service_instance(lower)
            , naming::id_type::unmanaged);

        lazy_results.push_back(
            stubs::primary_namespace::service_async<response>(
                target, req, action_priority_));
    }

    wait_all(lazy_results);

    BOOST_FOREACH(hpx::future<response> const& f, lazy_results)
    {
        response const& rep = f.get();
        if (success != rep.get_status())
        {
            HPX_THROWS_IF(ec, rep.get_status(),
                "addressing_service::send_refcnt_requests_sync",
                "could not decrement reference count");
            return;
        }
    }

    if (&ec != &throws)
        ec = make_success_code();
}

}}

