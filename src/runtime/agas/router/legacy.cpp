////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#if HPX_AGAS_VERSION > 0x10

#include <hpx/exception.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/agas/router/legacy.hpp>

namespace hpx { namespace agas
{

legacy_router::legacy_router(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
  , runtime_mode runtime_type_
):
    router_type(ini_.get_agas_router_mode())
  , runtime_type(runtime_type_)
  , state_(starting)
  , prefix_()
{ // {{{
    // boot the parcel port
    pp.run(false);

    create_big_boot_barrier(pp, ini_, runtime_type_);

    if (router_type == router_mode_bootstrap)
        launch_bootstrap(pp, ini_);
    else
        launch_hosted(pp, ini_);
} // }}}

void legacy_router::launch_bootstrap(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
) { // {{{
    using boost::asio::ip::address;

    bootstrap = boost::make_shared<bootstrap_data_type>();
    
    naming::locality l = ini_.get_agas_locality();

    address addr = address::from_string(l.get_address());
    endpoint_type ep(addr, l.get_port());

    gva_type primary_gva(ep,
        primary_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->primary_ns_server));
    gva_type component_gva(ep,
        component_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->component_ns_server));
    gva_type symbol_gva(ep,
        symbol_namespace_server_type::get_component_type(), 1U,
            static_cast<void*>(&bootstrap->symbol_ns_server));
    
    bootstrap->primary_ns_server.bind_locality(ep, 3);
    bootstrap->symbol_ns_server.bind("/locality(agas#0)",
        naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 

    local_prefix(naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX));

    bootstrap->primary_ns_server.bind_gid
        (primary_namespace_server_type::fixed_gid(), primary_gva);
    bootstrap->primary_ns_server.bind_gid
        (component_namespace_server_type::fixed_gid(), component_gva);
    bootstrap->primary_ns_server.bind_gid
        (symbol_namespace_server_type::fixed_gid(), symbol_gva);

    naming::gid_type lower, upper;
    get_id_range(l, HPX_INITIAL_GID_RANGE, lower, upper);
    get_runtime().get_id_pool().set_range(lower, upper);

    if (runtime_type == runtime_mode_console)
        bootstrap->symbol_ns_server.bind("/locality(console)",
            naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 

    get_big_boot_barrier().wait();

    state_.store(running);
} // }}}

void legacy_router::launch_hosted(
    parcelset::parcelport& pp 
  , util::runtime_configuration const& ini_
) { // {{{
    hosted = boost::make_shared<hosted_data_type>();

    hosted->gva_cache_.reserve(ini_.get_agas_gva_cache_size());
    hosted->locality_cache_.reserve(ini_.get_agas_locality_cache_size());

    get_big_boot_barrier().wait();

    state_.store(running);
} // }}}

bool legacy_router::get_prefix(
    naming::locality const& l
  , naming::gid_type& prefix
  , bool self
  , bool try_cache
  , error_code& ec
) { // {{{ get_prefix implementation
    using boost::asio::ip::address;
    using boost::fusion::at_c;

    if (try_cache && !is_bootstrap() &&
        get_prefix_cached(l, prefix, self, ec))
        return false;

    address addr = address::from_string(l.get_address());

    endpoint_type ep(addr, l.get_port()); 

    if (self)
    {
        response_type r;

        if (is_bootstrap())
            r = bootstrap->primary_ns_server.bind_locality(ep, 0);
        else
            r = hosted->primary_ns_.bind(ep, 0);

        naming::gid_type gid_ = naming::get_gid_from_prefix(r.get_prefix());

        if (gid_ != naming::invalid_gid)
        {
            prefix = gid_; 

            if (!is_bootstrap())
            {
                cache_mutex_type::scoped_lock
                    lock(hosted->locality_cache_mtx_);
                hosted->locality_cache_.insert(l, prefix);
            }
        }

        return r.get_status() == success;
    }
    
    else 
    {
        response_type r;

        if (is_bootstrap())
            r = bootstrap->primary_ns_server.resolve_locality(ep);
        else
            r = hosted->primary_ns_.resolve(ep); 

        naming::gid_type gid_ = naming::get_gid_from_prefix(r.get_prefix());

        if (gid_ != naming::invalid_gid)
        {
            prefix = gid_;

            if (!is_bootstrap())
            {
                cache_mutex_type::scoped_lock
                    lock(hosted->locality_cache_mtx_);
                hosted->locality_cache_.insert(l, prefix);
            }
        }

        return false;
    }
} // }}}

bool legacy_router::get_prefix_cached(
    naming::locality const& l
  , naming::gid_type& prefix
  , bool self
  , error_code& ec
) { // {{{
    if (is_bootstrap())
        return get_prefix(l, prefix, self, false, ec);

    locality_entry_type e;

    cache_mutex_type::scoped_lock
        lock(hosted->locality_cache_mtx_);

    if (hosted->locality_cache_.get_entry(l, e))
    {
        prefix = e.get();
        return true;
    }

    return false; 
} // }}}

bool legacy_router::remove_prefix(
    naming::locality const& l
  , error_code& ec
) { // {{{
    using boost::asio::ip::address;
    using boost::fusion::at_c;

    const address addr = address::from_string(l.get_address());

    const endpoint_type ep(addr, l.get_port()); 

    response_type r;

    if (is_bootstrap())
        r = bootstrap->primary_ns_server.unbind_locality(ep);
    else
        r = hosted->primary_ns_.unbind(ep);

    if (success == r.get_status())
        return true;
    else
        return false;
} // }}}

bool legacy_router::get_console_prefix(
    naming::gid_type& prefix
  , bool try_cache
  , error_code& ec
) { // {{{
    if (status() != running)
        return false;

    if (is_console())
    {
        prefix = local_prefix();
        return true;
    }
 
    if (try_cache && !is_bootstrap())
    {
        if (hosted->console_cache_)
        {
            prefix = naming::get_gid_from_prefix(hosted->console_cache_);
            return true;
        }
    }

    if (is_bootstrap()) {
        response_type r = 
            bootstrap->symbol_ns_server.resolve("/locality(console)");

        if ((r.get_gid() != naming::invalid_gid) &&
            (r.get_status() == success))
        {
            prefix = r.get_gid();
            return true;
        }
    }

    else {
        response_type r = hosted->symbol_ns_.resolve("/locality(console)");

        if ((r.get_gid() != naming::invalid_gid) &&
            (r.get_status() == success))
        {
            prefix = r.get_gid();
            hosted->console_cache_.store
                (naming::get_prefix_from_gid(prefix));
            return true;
        }
    }

    return false;
} // }}}

bool legacy_router::get_prefixes(
    std::vector<naming::gid_type>& prefixes
  , components::component_type type
  , error_code& ec 
) { // {{{ get_prefixes implementation
    if (type != components::component_invalid)
    {
        response_type r;

        if (is_bootstrap())
            r = bootstrap->component_ns_server.resolve_id
                (component_id_type(type));
        else
            r = hosted->component_ns_.resolve(component_id_type(type));

        const count_type s = r.get_localities_size();
        prefix_type* p = r.get_localities();

        // REVIEW: Check response status too? 
        if (!s)
            return false;

        for (count_type i = 0; i < s; ++i) 
            prefixes.push_back(naming::get_gid_from_prefix(p[i]));

        return true; 
    }

    else
    {
        response_type r;

        if (is_bootstrap())
            r = bootstrap->primary_ns_server.localities();
        else
            r = hosted->primary_ns_.localities();
       
        const count_type s = r.get_localities_size();
        prefix_type* p = r.get_localities();

        // REVIEW: Check response status too? 
        if (!s)
            return false;

        for (count_type i = 0; i < s; ++i) 
            prefixes.push_back(naming::get_gid_from_prefix(p[i]));
    
        return true;
    }
} // }}} 

components::component_type legacy_router::get_component_id(
    std::string const& name
  , error_code& ec 
) { /// {{{
    response_type r;

    if (is_bootstrap())
        r = bootstrap->component_ns_server.bind_name(name);
    else
        r = hosted->component_ns_.bind(name);

    // REVIEW: Check response status?
    return (components::component_type) r.get_component_type();
} // }}} 

components::component_type legacy_router::register_factory(
    naming::gid_type const& prefix
  , std::string const& name
  , error_code& ec
) { // {{{
    if (is_bootstrap())
        return (components::component_type)
            bootstrap->component_ns_server.bind_prefix(name,
                naming::get_prefix_from_gid(prefix)).get_component_type();
    else
        return (components::component_type)
            hosted->component_ns_.bind(name,
                naming::get_prefix_from_gid(prefix)).get_component_type();
} // }}}

bool legacy_router::get_id_range(
    naming::locality const& l
  , count_type count
  , naming::gid_type& lower_bound
  , naming::gid_type& upper_bound
  , error_code& ec
) { // {{{ get_id_range implementation
    using boost::asio::ip::address;
    using boost::fusion::at_c;

    address addr = address::from_string(l.get_address());

    endpoint_type ep(addr, l.get_port()); 
     
    response_type r;

    if (is_bootstrap())
        r = bootstrap->primary_ns_server.bind_locality(ep, count);

    // WARNING: this deadlocks if AGAS is unresponsive and all response
    // futures are checked out and pending.
    else
    {
        lcos::eager_future<
            primary_namespace_server_type::bind_locality_action,
            response_type
        >* f = 0;

        // get a future
        hosted->allocate_response_sema_.wait(1);
        hosted->allocate_response_pool_.dequeue(&f); 

        BOOST_ASSERT(f);

        // reset the future
        f->reset();

        // execute the action (synchronously)
        f->apply(
            naming::id_type(primary_namespace_server_type::fixed_gid()
                          , naming::id_type::unmanaged),
            ep, count);
        r = f->get();

        // return the future to the pool
        hosted->allocate_response_pool_.enqueue(f);
        hosted->allocate_response_sema_.signal(1);
    }

    lower_bound = r.get_lower_bound(); 
    upper_bound = r.get_upper_bound();

    return lower_bound && upper_bound;
} // }}}

bool legacy_router::bind_range(
    naming::gid_type const& lower_id
  , count_type count
  , naming::address const& baseaddr
  , offset_type offset
  , error_code& ec
) { // {{{ bind_range implementation
    using boost::asio::ip::address;
    using boost::fusion::at_c;

    address addr = address::from_string(baseaddr.locality_.get_address());

    endpoint_type ep(addr, baseaddr.locality_.get_port()); 
   
    // Create a global virtual address from the legacy calling convention
    // parameters.
    gva_type gva(ep, baseaddr.type_, count, baseaddr.address_, offset);
    
    if (is_bootstrap())
    {
        response_type r
            = bootstrap->primary_ns_server.bind_gid(lower_id, gva);

        if (success == r.get_status()) 
            return true;
    }

    else
    {
        // WARNING: this deadlocks if AGAS is unresponsive and all response
        // futures are checked out and pending.
        lcos::eager_future<
            primary_namespace_server_type::bind_gid_action,
            response_type
        >* f = 0;

        // get a future
        hosted->bind_response_sema_.wait(1);
        hosted->bind_response_pool_.dequeue(&f); 

        BOOST_ASSERT(f);

        // reset the future
        f->reset();

        // execute the action (synchronously)
        f->apply(
            naming::id_type(primary_namespace_server_type::fixed_gid()
                          , naming::id_type::unmanaged),
            lower_id, gva);
        response_type r = f->get();

        // return the future to the pool
        hosted->bind_response_pool_.enqueue(f);
        hosted->bind_response_sema_.signal(1);
     
        if (success == r.get_status()) 
        { 
            cache_mutex_type::scoped_lock
                lock(hosted->gva_cache_mtx_);
            gva_cache_key key(lower_id, count);
            hosted->gva_cache_.insert(key, gva);
            return true;
        }
    }

    return false; 
} // }}}

legacy_router::count_type legacy_router::incref(
    naming::gid_type const& id
  , count_type credits
  , error_code& ec 
) { // {{{
    if (is_bootstrap())
        return bootstrap->primary_ns_server.increment(id, credits).
            get_count();
    else
        return hosted->primary_ns_.increment(id, credits).get_count();
} // }}}

legacy_router::count_type legacy_router::decref(
    naming::gid_type const& id
  , components::component_type& t
  , count_type credits
  , error_code& ec
) { // {{{ decref implementation
    using boost::fusion::at_c;
    
    response_type r;
    
    if (is_bootstrap())
        r = bootstrap->primary_ns_server.decrement(id, credits);
    else
        r = hosted->primary_ns_.decrement(id, credits);

    if (0 == r.get_count()) {
        t = (components::component_type) r.get_component_type();
        BOOST_ASSERT(t != components::component_invalid);
    }

    return r.get_count();
} // }}}

bool legacy_router::unbind_range(
    naming::gid_type const& lower_id
  , count_type count
  , naming::address& addr
  , error_code& ec 
) { // {{{ unbind_range implementation
    response_type r;

    if (is_bootstrap())
        r = bootstrap->primary_ns_server.unbind_gid(lower_id, count);
    else
        r = hosted->primary_ns_.unbind(lower_id, count);

    if (!is_bootstrap() && (success == r.get_status()))
    {
        cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
        gva_erase_policy ep(lower_id, count);
        hosted->gva_cache_.erase(ep);
        addr.locality_ = r.get_gva().endpoint;
        addr.type_ = r.get_gva().type;
        addr.address_ = r.get_gva().lva();
    }

    return success == r.get_status(); 
} // }}}

bool legacy_router::resolve(
    naming::gid_type const& id
  , naming::address& addr
  , bool try_cache
  , error_code& ec
) { // {{{ resolve implementation
    if (!is_bootstrap())
    {
        // {{{ special cases: authoritative AGAS component address
        // resolution
        if (id == primary_namespace_server_type::fixed_gid())
        {
            addr =hosted->primary_ns_addr_;
            return true;
        }

        else if (id == component_namespace_server_type::fixed_gid())
        {
            addr = hosted->component_ns_addr_;
            return true;
        }

        else if (id == symbol_namespace_server_type::fixed_gid())
        {
            addr = hosted->symbol_ns_addr_;
            return true;
        }
        // }}}

        else if (try_cache && resolve_cached(id, addr, ec))
            return true;
    }
    
    response_type r; 

    if (is_bootstrap())
        r = bootstrap->primary_ns_server.resolve_gid(id);
    else
        r = hosted->primary_ns_.resolve(id);

    addr.locality_ = r.get_gva().endpoint;
    addr.type_ = r.get_gva().type;
    addr.address_ = r.get_gva().lva();

    if (success != r.get_status())
        return false;

    else if (is_bootstrap())
        return true;

    else
    {
        // We only insert the entry into the cache if it's valid
        cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
        gva_cache_key key(id);
        hosted->gva_cache_.insert(key, r.get_gva());
        return true;
    }
} // }}}

bool legacy_router::resolve_cached(
    naming::gid_type const& id
  , naming::address& addr
  , error_code& ec
) { // {{{ resolve_cached implementation
    // TODO: assert?
    if (is_bootstrap())
        return resolve(id, addr, false, ec);

    // first look up the requested item in the cache
    gva_cache_key k(id);
    cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
    gva_cache_key idbase;
    gva_cache_type::entry_type e;

    // Check if the entry is currently in the cache
    if (hosted->gva_cache_.get_entry(k, idbase, e))
    {
        if (HPX_UNLIKELY(id.get_msb() != idbase.id.get_msb()))
        {
            HPX_THROWS_IN_CURRENT_FUNC_IF(ec, bad_parameter, 
                "MSBs of GID base and GID do not match");
            return false;
        }

        gva_type const& gva = e.get();

        addr.locality_ = gva.endpoint;
        addr.type_ = gva.type;
        addr.address_ = gva.lva(id, idbase.id);
    
        return true;
    }

    return false;
} // }}}

bool legacy_router::registerid(
    std::string const& name
  , naming::gid_type const& id
  , error_code& ec
) { // {{{
    response_type r;

    if (is_bootstrap())
        r = bootstrap->symbol_ns_server.rebind(name, id);
    else
        r = hosted->symbol_ns_.rebind(name, id);

    return r.get_gid() == id;
} // }}}

bool legacy_router::unregisterid(
    std::string const& name
  , error_code& ec
) { // {{{
    response_type r;

    if (is_bootstrap())
        r = bootstrap->symbol_ns_server.unbind(name);
    else  
        r = hosted->symbol_ns_.unbind(name);

    return r.get_status() == success;
} // }}}

bool legacy_router::queryid(
    std::string const& ns_name
  , naming::gid_type& id
  , error_code& ec
) { // {{{
    response_type r;

    if (is_bootstrap())
        r = bootstrap->symbol_ns_server.resolve(ns_name);
    else
        r = hosted->symbol_ns_.resolve(ns_name);

    if (r.get_status() == success)
    {
        id = r.get_gid();
        return true;
    }

    else
        return false;
} // }}}

bool legacy_router::iterateids(
    iterateids_function_type const& f
) { // {{{
    response_type r;

    if (is_bootstrap())
        r = bootstrap->symbol_ns_server.iterate(f);
    else
        r = hosted->symbol_ns_.iterate(f);

    if (r.get_status() == success)
        return true;
    else
        return false;
} // }}}

}}

#endif // HPX_AGAS_VERSION > 0x10

