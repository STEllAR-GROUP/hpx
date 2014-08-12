////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
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
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/lcos/wait_all.hpp>
#if !defined(HPX_GCC44_WORKAROUND)
#include <hpx/lcos/broadcast.hpp>
#endif

#include <boost/format.hpp>
#include <boost/icl/closed_interval.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/vector.hpp>

namespace hpx { namespace detail
{
    std::string get_locality_base_name();
}}

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
        HPX_ASSERT(count_);
    }

    naming::gid_type get_gid() const
    {
        return boost::icl::lower(key_);
    }

    boost::uint64_t get_count() const
    {
        naming::gid_type const size = boost::icl::length(key_);
        HPX_ASSERT(size.get_msb() == 0);
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
            return boost::icl::contains(lhs.key_, rhs.key_);

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
  , console_cache_(naming::invalid_locality_id)
  , max_refcnt_requests_(ini_.get_agas_max_pending_refcnt_requests())
  , refcnt_requests_count_(0)
  , enable_refcnt_caching_(true)
  , refcnt_requests_(new refcnt_requests_type)
  , service_type(ini_.get_agas_service_mode())
  , runtime_type(runtime_type_)
  , caching_(ini_.get_agas_caching_mode())
  , range_caching_(caching_ ? ini_.get_agas_range_caching_mode() : false)
  , action_priority_(ini_.get_agas_dedicated_server() ?
        threads::thread_priority_normal : threads::thread_priority_boost)
  , here_()         // defer initializing this
  , rts_lva_(0)
  , mem_lva_(0)
  , state_(starting)
  , locality_()
{ // {{{
    create_big_boot_barrier(pp, ini_);

    if (caching_)
        gva_cache_->reserve(ini_.get_agas_local_cache_size());

    if (service_type == service_mode_bootstrap)
    {
        launch_bootstrap(pp, ini_);
    }
}

void addressing_service::initialize(parcelset::parcelport& pp)
{
    // now, boot the parcel port
    pp.run(false);

    if (service_type == service_mode_bootstrap)
    {
        get_big_boot_barrier().wait_bootstrap();
    }
    else
    {
        launch_hosted();
        get_big_boot_barrier().wait_hosted(pp.get_locality_name(),
            &hosted->primary_ns_server_, &hosted->symbol_ns_server_);
    }

    set_status(running);
} // }}}

naming::locality const& addressing_service::get_here() const
{
    if (!here_) {
        runtime* rt = get_runtime_ptr();
        HPX_ASSERT(rt &&
            rt->get_state() >= runtime::state_initialized &&
            rt->get_state() < runtime::state_stopped);
        here_ = rt->here();
    }
    return here_;
}

void* addressing_service::get_hosted_primary_ns_ptr() const
{
    HPX_ASSERT(0 != hosted.get());
    return &hosted->primary_ns_server_;
}

void* addressing_service::get_hosted_symbol_ns_ptr() const
{
    HPX_ASSERT(0 != hosted.get());
    return &hosted->symbol_ns_server_;
}

void* addressing_service::get_bootstrap_locality_ns_ptr() const
{
    HPX_ASSERT(0 != bootstrap.get());
    return &bootstrap->locality_ns_server_;
}

void* addressing_service::get_bootstrap_primary_ns_ptr() const
{
    HPX_ASSERT(0 != bootstrap.get());
    return &bootstrap->primary_ns_server_;
}

void* addressing_service::get_bootstrap_component_ns_ptr() const
{
    HPX_ASSERT(0 != bootstrap.get());
    return &bootstrap->component_ns_server_;
}

void* addressing_service::get_bootstrap_symbol_ns_ptr() const
{
    HPX_ASSERT(0 != bootstrap.get());
    return &bootstrap->symbol_ns_server_;
}

namespace detail
{
    boost::uint32_t get_number_of_pus_in_cores(boost::uint32_t num_cores);
}

void addressing_service::launch_bootstrap(
    parcelset::parcelport& pp
  , util::runtime_configuration const& ini_
    )
{ // {{{
    bootstrap = boost::make_shared<bootstrap_data_type>();

    runtime& rt = get_runtime();

    // store number of cores used by other processes
    boost::uint32_t cores_needed = rt.assign_cores();
    boost::uint32_t used_cores = rt.assign_cores(
        pp.get_locality_name(), cores_needed);
    util::runtime_configuration& cfg = rt.get_config();
    cfg.set_used_cores(used_cores);
    cfg.set_agas_locality(ini_.get_parcelport_address());
    rt.assign_cores();

    naming::locality const ep = ini_.get_parcelport_address();
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
    request locality_req(locality_ns_allocate, ep, 4, num_threads); //-V112
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
        request(primary_ns_bind_gid, locality_gid, locality_gva
          , naming::get_locality_id_from_gid(locality_gid))
      , request(primary_ns_bind_gid, primary_gid, primary_gva
          , naming::get_locality_id_from_gid(primary_gid))
      , request(primary_ns_bind_gid, component_gid, component_gva
          , naming::get_locality_id_from_gid(component_gid))
      , request(primary_ns_bind_gid, symbol_gid, symbol_gva
          , naming::get_locality_id_from_gid(symbol_gid))
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

            if (console_cache_ != naming::invalid_locality_id)
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
                if (console_cache_ == naming::invalid_locality_id) {
                    console_cache_ = console;
                }
                else {
                    HPX_ASSERT(console_cache_ == console);
                }
            }

            LAGAS_(debug) <<
                ( boost::format(
                  "addressing_server::get_console_locality, "
                  "caching console locality, prefix(%1%)")
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

bool addressing_service::bind_range_local(
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

        request req(primary_ns_bind_gid, lower_id, g,
            naming::get_locality_id_from_gid(lower_id));
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
            {
                // Put the range into the cache.
                update_cache_entry(lower_id, g, ec);
            }
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
        HPX_RETHROWS_IF(ec, e, "addressing_service::bind_range_local");
        return false;
    }
} // }}}

bool addressing_service::bind_postproc(
    future<response> f, naming::gid_type const& lower_id, gva const& g
    )
{
    response rep = f.get();
    error const s = rep.get_status();
    if (success != s && repeated_request != s)
        return false;

    if (caching_)
    {
        if (range_caching_)
            // Put the range into the cache.
            update_cache_entry(lower_id, g);

        else
        {
            // Only put the first GID in the range into the cache.
            gva const first_g = g.resolve(lower_id, lower_id);
            update_cache_entry(lower_id, first_g);
        }
    }

    return true;
}

hpx::future<bool> addressing_service::bind_range_async(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address const& baseaddr
  , boost::uint64_t offset
  , boost::uint32_t locality_id
    )
{
    // ask server
    naming::locality const& ep = baseaddr.locality_;

    // Create a global virtual address from the legacy calling convention
    // parameters.
    gva const g(ep, baseaddr.type_, count, baseaddr.address_, offset);

    naming::id_type target(
        stubs::primary_namespace::get_service_instance(lower_id)
      , naming::id_type::unmanaged);

    request req(primary_ns_bind_gid, lower_id, g, locality_id);
    response rep;

    using util::placeholders::_1;
    future<response> f =
        stubs::primary_namespace::service_async<response>(target, req);
    return f.then(
        util::bind(&addressing_service::bind_postproc, this, _1, lower_id, g)
    );
}

bool addressing_service::unbind_range_local(
    naming::gid_type const& lower_id
  , boost::uint64_t count
  , naming::address& addr
  , error_code& ec
    )
{ // {{{ unbind_range implementation
    try {
        request req(primary_ns_unbind_gid, lower_id, count);
        response rep;

//         if (get_status() == running &&
//             naming::get_locality_id_from_gid(lower_id) !=
//                 naming::get_locality_id_from_gid(locality_))
//         {
//             naming::id_type target(
//                 stubs::primary_namespace::get_service_instance(lower_id)
//               , naming::id_type::unmanaged);
//
//             rep = stubs::primary_namespace::service(
//                 target, req, threads::thread_priority_default, ec);
//         }
//         else

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
        HPX_RETHROWS_IF(ec, e, "addressing_service::unbind_range_local");
        return false;
    }
} // }}}

/// This function will test whether the given address refers to an object
/// living on the locality of the caller. We rely completely on the local AGAS
/// cache and local AGAS instance, assuming that everything which is not in
/// the cache is not local.

// bool addressing_service::is_local_address(
//     naming::gid_type const& id
//   , naming::address& addr
//   , error_code& ec
//     )
// {
//     // Resolve the address of the GID.
//
//     // NOTE: We do not throw here for a reason; it is perfectly valid for the
//     // GID to not be found in the local AGAS instance.
//     if (!resolve(id, addr, ec) || ec)
//         return false;
//
//     return addr.locality_ == get_here();
// }

bool addressing_service::is_local_address_cached(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
    )
{
    // Try to resolve the address of the GID from the locally available
    // information.

    // NOTE: We do not throw here for a reason; it is perfectly valid for the
    // GID to not be found in the cache.
    if (!resolve_cached(id, addr, ec) || ec)
    {
        if (ec) return false;

        // try also the local part of AGAS before giving up
        if (!resolve_full_local(id, addr, ec) || ec)
            return false;
    }

    return addr.locality_ == get_here();
}

// Return true if at least one address is local.
// bool addressing_service::is_local_address(
//     naming::gid_type const* gids
//   , naming::address* addrs
//   , std::size_t size
//   , boost::dynamic_bitset<>& locals
//   , error_code& ec
//     )
// {
//     // Try the cache
//     if (caching_)
//     {
//         bool all_resolved = resolve_cached(gids, addrs, size, locals, ec);
//         if (ec)
//             return false;
//         if (all_resolved)
//             return locals.any();      // all destinations resolved
//     }
//
//     if (!resolve_full(gids, addrs, size, locals, ec) || ec)
//         return false;
//
//     return locals.any();
// }

bool addressing_service::is_local_lva_encoded_address(
    boost::uint64_t msb
    )
{
    // NOTE: This should still be migration safe.
    return naming::detail::strip_internal_bits_from_gid(msb) ==
        get_local_locality().get_msb();
}

bool addressing_service::resolve_locally_known_addresses(
    naming::gid_type const& id
  , naming::address& addr
    )
{
    // LVA-encoded GIDs (located on this machine)
    boost::uint64_t lsb = id.get_lsb();
    boost::uint64_t msb = naming::detail::strip_internal_bits_from_gid(id.get_msb());

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
            if (!mem_lva_)
                mem_lva_ = get_runtime().get_memory_lva();

            addr.type_ = components::component_memory;
            addr.address_ = mem_lva_;
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

bool addressing_service::resolve_full_local(
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
        naming::gid_type base_gid = rep.get_base_gid();
        gva const base_gva = rep.get_gva();

        gva const g = base_gva.resolve(id, base_gid);

        addr.locality_ = g.endpoint;
        addr.type_ = g.type;
        addr.address_ = g.lva();

        if (caching_)
        {
            if (range_caching_)
                // Put the gva range into the cache.
                update_cache_entry(base_gid, base_gva, ec);
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
        HPX_RETHROWS_IF(ec, e, "addressing_service::resolve_full_local");
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
        const boost::uint64_t id_msb =
            naming::detail::strip_internal_bits_from_gid(id.get_msb());

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

/*
        LAGAS_(debug) <<
            ( boost::format(
                "addressing_service::resolve_cached, "
                "cache hit for address %1%, lva %2% (base %3%, lva %4%)")
            % id
            % reinterpret_cast<void*>(addr.address_)
            % idbase.get_gid()
            % reinterpret_cast<void*>(g.lva()));
*/

        return true;
    }

    if (&ec != &throws)
        ec = make_success_code();

    LAGAS_(debug) <<
        ( boost::format(
            "addressing_service::resolve_cached, "
            "cache miss for address %1%")
        % id);

    return false;
} // }}}

hpx::future<naming::address> addressing_service::resolve_async(
    naming::gid_type const& gid
    )
{
    if (!gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_async",
            "invalid reference id");
        return make_ready_future(naming::address());
    }

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

hpx::future<naming::id_type> addressing_service::get_colocation_id_async(
    naming::id_type const& id
    )
{
    if (!id)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::get_colocation_id_async",
            "invalid reference id");
        return make_ready_future(naming::invalid_id);
    }

    agas::request req(agas::primary_ns_resolve_gid, id.get_gid());
    naming::id_type service_target(
        agas::stubs::primary_namespace::get_service_instance(id.get_gid())
        , naming::id_type::unmanaged);

    return stubs::primary_namespace::service_async<naming::id_type>(
        service_target, req);
}

///////////////////////////////////////////////////////////////////////////////
naming::address addressing_service::resolve_full_postproc(
    future<response> f, naming::gid_type const& id
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
    naming::gid_type base_gid = rep.get_base_gid();
    gva const base_gva = rep.get_gva();

    gva const g = base_gva.resolve(id, base_gid);

    addr.locality_ = g.endpoint;
    addr.type_ = g.type;
    addr.address_ = g.lva();

    if (caching_)
    {
        if (range_caching_)
            // Put the gva range into the cache.
            update_cache_entry(base_gid, base_gva);
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
    if (!gid)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::resolve_full_async",
            "invalid reference id");
        return make_ready_future(naming::address());
    }

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
    future<response> f =
        stubs::primary_namespace::service_async<response>(target, req);
    return f.then(
        util::bind(&addressing_service::resolve_full_postproc, this, _1, gid)
    );
}

///////////////////////////////////////////////////////////////////////////////
bool addressing_service::resolve_full_local(
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

            HPX_ASSERT(j < reps.size());
            if (success != reps[j].get_status())
                return false;

            // Resolve the gva to the real resolved address (which is just a gva
            // with as fully resolved LVA and an offset of zero).
            naming::gid_type base_gid = reps[j].get_base_gid();
            gva const base_gva = reps[j].get_gva();

            gva const g = base_gva.resolve(gids[i], base_gid);

            naming::address& addr = addrs[i];
            addr.locality_ = g.endpoint;
            addr.type_ = g.type;
            addr.address_ = g.lva();

            if (caching_)
            {
                if (range_caching_) {
                    // Put the gva range into the cache.
                    update_cache_entry(base_gid, base_gva, ec);
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
    naming::id_type const* ids = p.get_destinations();

    naming::id_type const target(
        stubs::primary_namespace::get_service_instance(ids[0])
      , naming::id_type::unmanaged);

    typedef server::primary_namespace::service_action action_type;

    // Determine whether the gid is local or remote
    naming::address addr;
    if (is_local_address_cached(target.get_gid(), addr))
    {
        // route through the local AGAS service instance
        applier::detail::apply_l_p<action_type>(
            target, addr, action_priority_, req);
        f(boost::system::error_code(), 0);      // invoke callback
        return;
    }

    // apply remotely, route through main AGAS service if the destination is
    // a service instance
    if (!addr)
    {
        if (stubs::primary_namespace::is_service_instance(ids[0]) ||
            stubs::symbol_namespace::is_service_instance(ids[0]))
        {
            // construct wrapper parcel
            naming::id_type const route_target(
                bootstrap_primary_namespace_gid(), naming::id_type::unmanaged);

            parcelset::parcel route_p(route_target, primary_ns_addr_
              , new hpx::actions::transfer_action<action_type>(action_priority_,
                    util::forward_as_tuple(req)));

            // send to the main AGAS instance for routing
            hpx::applier::get_applier().get_parcel_handler().put_parcel(route_p, f);
            return;
        }
    }

    // apply directly as we have the resolved destination address
    applier::detail::apply_r_p_cb<action_type>(std::move(addr), target,
        action_priority_, f, req);
}

///////////////////////////////////////////////////////////////////////////////
// The parameter 'compensated_credit' holds the amount of credits to be added
// to the acknowledged number of credits. The compensated credits are non-zero
// if there was a pending decref request at the point when the incref was sent.
// The pending decref was subtracted from the amount of credits to incref.
boost::int64_t addressing_service::synchronize_with_async_incref(
    hpx::future<boost::int64_t> fut
  , naming::id_type const& id
  , boost::int64_t compensated_credit
    )
{
    return fut.get() + compensated_credit;
}

lcos::future<boost::int64_t> addressing_service::incref_async(
    naming::gid_type const& gid
  , boost::int64_t credit
  , naming::id_type const& keep_alive
    )
{ // {{{ incref implementation
    if (HPX_UNLIKELY(0 == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        lcos::future<boost::int64_t> (
                addressing_service::*incref_async_ptr)(
            naming::gid_type const&
          , boost::int64_t
          , naming::id_type const&
        ) = &addressing_service::incref_async;

        return async(incref_async_ptr, this, gid, credit, keep_alive);
    }

    if (HPX_UNLIKELY(0 >= credit))
    {
        HPX_THROW_EXCEPTION(bad_parameter
          , "addressing_service::incref_async"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return lcos::future<boost::int64_t>();
    }

    HPX_ASSERT(keep_alive != naming::invalid_id);

    typedef refcnt_requests_type::value_type mapping;

    // Some examples of calculating the compensated credits below
    //
    //  case   pending   credits   remaining   sent to   compensated
    //  no     decref              decrefs     AGAS      credits
    // ------+---------+---------+------------+--------+-------------
    //   1         0        10        0           0        10
    //   2        10         9        1           0        10
    //   3        10        10        0           0        10
    //   4        10        11        0           1        10

    mapping pending_incref;
    bool has_pending_incref = false;
    boost::int64_t pending_decrefs = 0;

    {
        mutex_type::scoped_lock l(refcnt_requests_mtx_);

        typedef refcnt_requests_type::key_type key_type;
        typedef refcnt_requests_type::data_type data_type;
        typedef refcnt_requests_type::iterator iterator;

        naming::gid_type raw = naming::detail::get_stripped_gid(gid);

        iterator matches = refcnt_requests_->find(raw);
        if (matches != refcnt_requests_->end())
            pending_decrefs = matches->data_;

        refcnt_requests_->apply(raw, util::incrementer<boost::int64_t>(credit));

        // Collect all entries which require to increment the refcnt, those
        // need to be handled immediately.
        matches = refcnt_requests_->find(raw);
        if (matches != refcnt_requests_->end())
        {
            data_type& match_data = matches->data_;

            // If the given incref was fully compensated by a pending decref
            // (i.e. match_data is less than 0) then there is no need
            // to do anything more.
            if (match_data > 0)
            {
                // credit > decrefs (case no 4): store the remaining incref to
                // be handled below.
                pending_incref = mapping(matches->key_, match_data);
                has_pending_incref = true;

                refcnt_requests_->erase(matches);
            }
            else if (match_data == 0)
            {
                // credit == decref (case no. 3): if the incref offsets any
                // pending decref, just remove the pending decref request.
                refcnt_requests_->erase(matches);
            }
            else
            {
                // credit < decref (case no. 2): do nothing
            }
        }
        else
        {
            // case no. 1
            HPX_ASSERT(pending_decrefs == 0);
            HPX_ASSERT(!has_pending_incref);

            // we pass the credits to the pre-resolved callback below
            pending_decrefs = credit;
        }
    }

    if (!has_pending_incref)
    {
        // no need to talk to AGAS, acknowledge the incref immediately
        return hpx::make_ready_future(pending_decrefs);
    }

    naming::gid_type const e_lower = boost::icl::lower(pending_incref.key());
    request req(primary_ns_increment_credit, e_lower, pending_incref.data());

    naming::id_type target(
        stubs::primary_namespace::get_service_instance(e_lower)
      , naming::id_type::unmanaged);

    lcos::future<boost::int64_t> f =
        stubs::primary_namespace::service_async<boost::int64_t>(target, req);

    // pass the amount of compensated decrefs to the callback
    using util::placeholders::_1;
    return f.then(
        util::bind(&addressing_service::synchronize_with_async_incref,
            this, _1, keep_alive, pending_decrefs));
} // }}}

///////////////////////////////////////////////////////////////////////////////
void addressing_service::decref(
    naming::gid_type const& gid
  , boost::int64_t credit
  , error_code& ec
    )
{ // {{{ decref implementation
    if (HPX_UNLIKELY(0 == threads::get_self_ptr()))
    {
        // reschedule this call as an HPX thread
        void (addressing_service::*decref_ptr)(
            naming::gid_type const&
          , boost::int64_t
          , error_code&
        ) = &addressing_service::decref;

        threads::register_thread_nullary(
            util::bind(decref_ptr, this, gid, credit, boost::ref(throws)),
            "addressing_service::decref");
        return;
    }

    if (HPX_UNLIKELY(credit <= 0))
    {
        HPX_THROWS_IF(ec, bad_parameter
          , "addressing_service::decref"
          , boost::str(boost::format("invalid credit count of %1%") % credit));
        return;
    }

    try {
        naming::gid_type raw = naming::detail::get_stripped_gid(gid);
        mutex_type::scoped_lock l(refcnt_requests_mtx_);

        // Match the decref request with entries in the incref table
        refcnt_requests_->apply(raw, util::decrementer<boost::int64_t>(credit));
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

static bool correct_credit_on_failure(future<bool> f, naming::id_type id,
    boost::int64_t mutable_gid_credit, boost::int64_t new_gid_credit)
{
    // Return the credit to the GID if the operation failed
    if (f.has_exception() && mutable_gid_credit != 0)
    {
        naming::detail::add_credit_to_gid(id.get_gid(), new_gid_credit);
        return false;
    }
    return true;
}

lcos::future<bool> addressing_service::register_name_async(
    std::string const& name
  , naming::id_type const& id
    )
{ // {{{
    // We need to modify the reference count.
    naming::gid_type& mutable_gid = const_cast<naming::id_type&>(id).get_gid();
    naming::gid_type new_gid = naming::detail::split_gid_if_needed(mutable_gid);

    request req(symbol_ns_bind, name, new_gid);

    future<bool> f = stubs::symbol_namespace::service_async<bool>(
        name, req, action_priority_);

    boost::int64_t new_credit = naming::detail::get_credit_from_gid(new_gid);
    if (new_credit != 0)
    {
        using util::placeholders::_1;
        return f.then(
            util::bind(correct_credit_on_failure, _1, id,
                HPX_GLOBALCREDIT_INITIAL, new_credit)
        );
    }

    return std::move(f);
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

namespace detail
{
    hpx::future<hpx::id_type> on_register_event(hpx::future<bool> f,
        lcos::promise<hpx::id_type, naming::gid_type> p)
    {
        if (!f.get())
        {
            HPX_THROW_EXCEPTION(bad_request,
                "hpx::agas::detail::on_register_event",
                "request 'symbol_ns_on_event' failed");
            return hpx::future<hpx::id_type>();
        }
        return p.get_future();
    }
}

future<hpx::id_type> addressing_service::on_symbol_namespace_event(
    std::string const& name, namespace_action_code evt,
    bool call_for_past_events)
{
    if (evt != symbol_ns_bind)
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "addressing_service::on_symbol_namespace_event",
            "invalid event type");
        return hpx::future<hpx::id_type>();
    }

    lcos::promise<naming::id_type, naming::gid_type> p;
    request req(symbol_ns_on_event, name, evt, call_for_past_events, p.get_gid());
    hpx::future<bool> f = stubs::symbol_namespace::service_async<bool>(
        name, req, action_priority_);

    using util::placeholders::_1;
    return f.then(util::bind(&detail::on_register_event, _1, std::move(p)));
}

}}

#if !defined(HPX_GCC44_WORKAROUND)

///////////////////////////////////////////////////////////////////////////////
typedef hpx::agas::server::symbol_namespace::service_action
    symbol_namespace_service_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(symbol_namespace_service_action)
HPX_REGISTER_BROADCAST_ACTION(symbol_namespace_service_action)

#endif

namespace hpx { namespace agas
{
    namespace detail
    {
        std::vector<hpx::id_type> find_all_symbol_namespace_services()
        {
            std::vector<hpx::id_type> ids;
            BOOST_FOREACH(hpx::id_type const& id, hpx::find_all_localities())
            {
                ids.push_back(hpx::id_type(
                    agas::stubs::symbol_namespace::get_service_instance(id),
                    id_type::unmanaged));
            }
            return ids;
        }
    }

/// Invoke the supplied hpx::function for every registered global name
bool addressing_service::iterate_ids(
    iterate_names_function_type const& f
  , error_code& ec
    )
{ // {{{
    try {
        request req(symbol_ns_iterate_names, f);

#if !defined(HPX_GCC44_WORKAROUND)
        symbol_namespace_service_action act;
        lcos::broadcast(act, detail::find_all_symbol_namespace_services(), req).get(ec);
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

        LAGAS_(debug) <<
            ( boost::format(
            "addressing_service::insert_cache_entry, gid(%1%), count(%2%)")
            % gid % count);

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        const gva_cache_key key(gid, count);

        if (!gva_cache_->insert(key, g))
        {
            // Figure out who we collided with.
            gva_cache_key idbase;
            gva_cache_type::entry_type e;

            if (!gva_cache_->get_entry(key, idbase, e))
            {
                // This is impossible under sane conditions.
                HPX_THROWS_IF(ec, invalid_data
                  , "addressing_service::insert_cache_entry"
                  , "data corruption or lock error occurred in cache");
                return;
            }

            LAGAS_(warning) <<
                ( boost::format(
                    "addressing_service::insert_cache_entry, "
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

    if (naming::get_locality_id_from_gid(gid) ==
        naming::get_locality_id_from_gid(locality_))
    {
        // we prefer not to store any local items in the AGAS cache
        return;
    }

    try {
        // The entry in AGAS for a locality's RTS component has a count of 0,
        // so we convert it to 1 here so that the cache doesn't break.
        const boost::uint64_t count = (g.count ? g.count : 1);

        LAGAS_(debug) <<
            ( boost::format(
            "addressing_service::update_cache_entry, gid(%1%), count(%2%)"
            ) % gid % count);

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        // update cache only if it's currently not locked
//         cache_mutex_type::scoped_try_lock lock(gva_cache_mtx_);
//         if (!lock)
//         {
//             if (&ec != &throws)
//                 ec = make_success_code();
//             return;
//         }

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

            LAGAS_(warning) <<
                ( boost::format(
                    "addressing_service::update_cache_entry, "
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
        LAGAS_(warning) << "addressing_service::clear_cache, clearing cache";

        cache_mutex_type::scoped_lock lock(gva_cache_mtx_);

        gva_cache_->clear();

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e, "addressing_service::clear_cache");
    }
} // }}}

// Disable refcnt caching during shutdown
void addressing_service::start_shutdown(error_code& ec)
{
    // If caching is disabled, we silently pretend success.
    if (!caching_)
        return;

    mutex_type::scoped_lock l(refcnt_requests_mtx_);
    enable_refcnt_caching_ = false;
    send_refcnt_requests_sync(l, ec);
}

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

///////////////////////////////////////////////////////////////////////////////
std::size_t addressing_service::get_cache_get_entry_count(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_get_entry_count(reset);
}

std::size_t addressing_service::get_cache_insert_entry_count(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_insert_entry_count(reset);
}

std::size_t addressing_service::get_cache_update_entry_count(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_update_entry_count(reset);
}

std::size_t addressing_service::get_cache_erase_entry_count(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_erase_entry_count(reset);
}

std::size_t addressing_service::get_cache_get_entry_time(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_get_entry_time(reset);
}

std::size_t addressing_service::get_cache_insert_entry_time(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_insert_entry_time(reset);
}

std::size_t addressing_service::get_cache_update_entry_time(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_update_entry_time(reset);
}

std::size_t addressing_service::get_cache_erase_entry_time(bool reset)
{
    cache_mutex_type::scoped_lock lock(gva_cache_mtx_);
    return gva_cache_->get_statistics().get_erase_entry_time(reset);
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

    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_get_entry_count(
        boost::bind(&addressing_service::get_cache_get_entry_count, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_insert_entry_count(
        boost::bind(&addressing_service::get_cache_insert_entry_count, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_update_entry_count(
        boost::bind(&addressing_service::get_cache_update_entry_count, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_erase_entry_count(
        boost::bind(&addressing_service::get_cache_erase_entry_count, this, ::_1));

    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_get_entry_time(
        boost::bind(&addressing_service::get_cache_get_entry_time, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_insert_entry_time(
        boost::bind(&addressing_service::get_cache_insert_entry_time, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_update_entry_time(
        boost::bind(&addressing_service::get_cache_update_entry_time, this, ::_1));
    HPX_STD_FUNCTION<boost::int64_t(bool)> cache_erase_entry_time(
        boost::bind(&addressing_service::get_cache_erase_entry_time, this, ::_1));

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
        },

        { "/agas/count/cache_get_entry", performance_counters::counter_raw,
          "returns the number of invocations of get_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_get_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache_insert_entry", performance_counters::counter_raw,
          "returns the number of invocations of insert_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insert_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache_update_entry", performance_counters::counter_raw,
          "returns the number of invocations of update_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_update_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },
        { "/agas/count/cache_erase_entry", performance_counters::counter_raw,
          "returns the number of invocations of erase_entry function of the "
                "AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_erase_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          ""
        },

        { "/agas/time/cache_get_entry", performance_counters::counter_raw,
          "returns the the overall time spent executing of the get_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_get_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
        },
        { "/agas/time/cache_insert_entry", performance_counters::counter_raw,
          "returns the the overall time spent executing of the insert_entry API "
              "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_insert_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
        },
        { "/agas/time/cache_update_entry", performance_counters::counter_raw,
          "returns the the overall time spent executing of the update_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_update_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
        },
        { "/agas/time/cache_erase_entry", performance_counters::counter_raw,
          "returns the the overall time spent executing of the erase_entry API "
                "function of the AGAS cache",
          HPX_PERFORMANCE_COUNTER_V1,
          boost::bind(&performance_counters::locality_raw_counter_creator,
              _1, cache_erase_entry_count, _2),
          &performance_counters::locality_counter_discoverer,
          "ns"
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
    mutex_type::scoped_lock l(refcnt_requests_mtx_, boost::try_to_lock);
    if (!l) return;     // no need to compete for garbage collection

    send_refcnt_requests_non_blocking(l, ec);
}

void addressing_service::garbage_collect(
    error_code& ec
    )
{
    mutex_type::scoped_lock l(refcnt_requests_mtx_, boost::try_to_lock);
    if (!l) return;     // no need to compete for garbage collection

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

    if (!enable_refcnt_caching_ || max_refcnt_requests_ == ++refcnt_requests_count_)
        send_refcnt_requests_non_blocking(l, ec);

    else if (&ec != &throws)
        ec = make_success_code();
}

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
    void dump_refcnt_requests(
        addressing_service::mutex_type::scoped_lock& l
      , addressing_service::refcnt_requests_type const& requests
      , const char* func_name
        )
    {
        std::stringstream ss;
        ss << ( boost::format(
              "%1%, dumping client-side refcnt table, requests(%2%):")
              % func_name % requests.size());

        typedef addressing_service::refcnt_requests_type::const_reference
            const_reference;

        BOOST_FOREACH(const_reference e, requests)
        {
            naming::gid_type const lower = boost::icl::lower(e.key());
            naming::gid_type const upper = boost::icl::upper(e.key());

            naming::gid_type const length_gid = (upper - lower);
            HPX_ASSERT(length_gid.get_msb() == 0);
            boost::uint64_t const length = length_gid.get_lsb() + 1;

            // The [client] tag is in there to make it easier to filter
            // through the logs.
            ss << ( boost::format(
                  "\n  [client] lower(%1%), upper(%2%), length(%3%), "
                  "credits(%4%)")
                  % lower
                  % upper
                  % length
                  % e.data());
        }

        LAGAS_(debug) << ss.str();
    }
#endif

void addressing_service::send_refcnt_requests_non_blocking(
    addressing_service::mutex_type::scoped_lock& l
  , error_code& ec
    )
{
    try {
        if (refcnt_requests_->empty())
        {
            l.unlock();
            return;
        }

        boost::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

        p.swap(refcnt_requests_);
        refcnt_requests_count_ = 0;

        l.unlock();

        LAGAS_(info) << (boost::format(
            "addressing_service::send_refcnt_requests_non_blocking, "
            "requests(%1%)")
            % p->size());

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
        if (LAGAS_ENABLED(debug))
            dump_refcnt_requests(l, *p,
                "addressing_service::send_refcnt_requests_non_blocking");
#endif

        // collect all requests for each locality
        typedef std::map<naming::id_type, std::vector<request> > requests_type;
        requests_type requests;

        BOOST_FOREACH(refcnt_requests_type::const_reference e, *p)
        {
            HPX_ASSERT(e.data() < 0);

            naming::gid_type lower(boost::icl::lower(e.key()));
            request const req(primary_ns_decrement_credit
              , lower, boost::icl::upper(e.key()), e.data());

            naming::id_type target(
                stubs::primary_namespace::get_service_instance(lower)
              , naming::id_type::unmanaged);

            requests[target].push_back(req);
        }

        // send requests to all locality
        requests_type::const_iterator end = requests.end();
        for (requests_type::const_iterator it = requests.begin(); it != end; ++it)
        {
            stubs::primary_namespace::bulk_service_non_blocking(
                (*it).first, (*it).second, action_priority_);
        }

        if (&ec != &throws)
            ec = make_success_code();
    }
    catch (hpx::exception const& e) {
        HPX_RETHROWS_IF(ec, e,
            "addressing_service::send_refcnt_requests_non_blocking");
    }
}

std::vector<hpx::future<std::vector<response> > >
addressing_service::send_refcnt_requests_async(
    addressing_service::mutex_type::scoped_lock& l
    )
{
    if (refcnt_requests_->empty())
    {
        l.unlock();
        return std::vector<hpx::future<std::vector<response> > >();
    }

    boost::shared_ptr<refcnt_requests_type> p(new refcnt_requests_type);

    p.swap(refcnt_requests_);
    refcnt_requests_count_ = 0;

    l.unlock();

    LAGAS_(info) << (boost::format(
        "addressing_service::send_refcnt_requests_async, "
        "requests(%1%)")
        % p->size());

#if defined(HPX_AGAS_DUMP_REFCNT_ENTRIES)
    if (LAGAS_ENABLED(debug))
        dump_refcnt_requests(l, *p,
            "addressing_service::send_refcnt_requests_sync");
#endif

    // collect all requests for each locality
    typedef std::map<naming::id_type, std::vector<request> > requests_type;
    requests_type requests;

    std::vector<hpx::future<std::vector<response> > > lazy_results;
    BOOST_FOREACH(refcnt_requests_type::const_reference e, *p)
    {
        HPX_ASSERT(e.data() < 0);

        naming::gid_type lower(boost::icl::lower(e.key()));
        request const req(primary_ns_decrement_credit
                        , lower
                        , boost::icl::upper(e.key())
                        , e.data());

        naming::id_type target(
            stubs::primary_namespace::get_service_instance(lower)
          , naming::id_type::unmanaged);

        requests[target].push_back(req);
    }

    // send requests to all locality
    requests_type::const_iterator end = requests.end();
    for (requests_type::const_iterator it = requests.begin(); it != end; ++it)
    {
        lazy_results.push_back(
            stubs::primary_namespace::bulk_service_async(
                (*it).first, (*it).second, action_priority_));
    }

    return lazy_results;
}

void addressing_service::send_refcnt_requests_sync(
    addressing_service::mutex_type::scoped_lock& l
  , error_code& ec
    )
{
    std::vector<hpx::future<std::vector<response> > > lazy_results =
        send_refcnt_requests_async(l);

    wait_all(lazy_results);

    BOOST_FOREACH(hpx::future<std::vector<response> > & f, lazy_results)
    {
        std::vector<response> const& reps = f.get();
        BOOST_FOREACH(response const& rep, reps)
        {
            if (success != rep.get_status())
            {
                HPX_THROWS_IF(ec, rep.get_status(),
                    "addressing_service::send_refcnt_requests_sync",
                    "could not decrement reference count (reported error" +
                    hpx::get_error_what(ec) + ", " +
                    hpx::get_error_file_name(ec) + "(" +
                    boost::lexical_cast<std::string>(
                        hpx::get_error_line_number(ec)) + "))");
                return;
            }
        }
    }

    if (&ec != &throws)
        ec = make_success_code();
}

}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        inline std::string name_from_basename(char const* basename, std::size_t idx)
        {
            std::string name;

            if (basename[0] != '/')
                name += '/';
            name += basename;
            if (name[name.size()-1] != '/')
                name += '/';
            name += boost::lexical_cast<std::string>(idx);

            return name;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<hpx::id_type> >
        find_all_ids_from_basename(char const* basename, std::size_t num_ids)
    {
        if (0 == basename)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_all_ids_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type> > results;
        for(std::size_t i = 0; i != num_ids; ++i)
        {
            std::string name = detail::name_from_basename(basename, i);
            results.push_back(agas::on_symbol_namespace_event(
                name, agas::symbol_ns_bind, true));
        }
        return results;
    }

    std::vector<hpx::future<hpx::id_type> >
        find_ids_from_basename(char const* basename,
            std::vector<std::size_t> const& ids)
    {
        if (0 == basename)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_ids_from_basename",
                "no basename specified");
        }

        std::vector<hpx::future<hpx::id_type> > results;
        BOOST_FOREACH(std::size_t i, ids)
        {
            std::string name = detail::name_from_basename(basename, i);
            results.push_back(agas::on_symbol_namespace_event(
                name, agas::symbol_ns_bind, true));
        }
        return results;
    }

    hpx::future<hpx::id_type> find_id_from_basename(char const* basename,
        std::size_t sequence_nr)
    {
        if (0 == basename)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::find_id_from_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::on_symbol_namespace_event(name, agas::symbol_ns_bind, true);
    }

    hpx::future<bool> register_id_with_basename(char const* basename,
        hpx::id_type id, std::size_t sequence_nr)
    {
        if (0 == basename)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::register_id_with_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::register_name(name, id);
    }

    hpx::future<hpx::id_type> unregister_id_with_basename(
        char const* basename, std::size_t sequence_nr)
    {
        if (0 == basename)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "hpx::unregister_id_with_basename",
                "no basename specified");
        }

        if (sequence_nr == std::size_t(~0U))
            sequence_nr = std::size_t(naming::get_locality_id_from_id(find_here()));

        std::string name = detail::name_from_basename(basename, sequence_nr);
        return agas::unregister_name(name);
    }
}
