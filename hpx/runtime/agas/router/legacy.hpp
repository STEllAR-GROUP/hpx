////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_15D904C7_CD18_46E1_A54A_65059966A34F)
#define HPX_15D904C7_CD18_46E1_A54A_65059966A34F

#include <hpx/config.hpp>

#include <vector>

#include <boost/atomic.hpp>
#include <boost/make_shared.hpp>
#include <boost/cache/entries/lfu_entry.hpp>
#include <boost/cache/local_cache.hpp>
#include <boost/cstdint.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/noncopyable.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/network/gva.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/runtime_configuration.hpp>

namespace hpx { namespace agas
{

// TODO: pass error codes once they're implemented in AGAS.
template <typename Database>
struct legacy_router : boost::noncopyable
{
    // {{{ types 
    typedef primary_namespace<Database, tag::network::tcpip>
        primary_namespace_type;

    typedef component_namespace<Database> component_namespace_type;
    typedef symbol_namespace<Database> symbol_namespace_type;

    typedef typename primary_namespace_type::server_type
        primary_namespace_server_type; 

    typedef typename component_namespace_type::server_type
        component_namespace_server_type; 

    typedef typename symbol_namespace_type::server_type
        symbol_namespace_server_type; 

    typedef typename component_namespace_type::component_id_type
        component_id_type;

    typedef typename primary_namespace_type::gva_type gva_type;
    typedef typename primary_namespace_type::count_type count_type;
    typedef typename primary_namespace_type::offset_type offset_type;
    typedef typename primary_namespace_type::endpoint_type endpoint_type;
    typedef typename primary_namespace_type::unbinding_type unbinding_type;
    typedef typename primary_namespace_type::binding_type binding_type;
    typedef typename component_namespace_type::prefixes_type prefixes_type;
    typedef typename component_namespace_type::prefix_type prefix_type;
    typedef typename primary_namespace_type::decrement_type decrement_type;

    typedef hpx::lcos::mutex cache_mutex_type;

    typedef boost::atomic<boost::uint32_t> console_cache_type;
    // }}}

    // {{{ gva cache
    struct gva_cache_key
    { // {{{ gva_cache_key implementation
        gva_cache_key()
          : id(), count(0)
        {}

        explicit gva_cache_key(naming::gid_type const& id_,
                               count_type count_ = 1)
          : id(id_), count(count_)
        {}

        naming::gid_type id;
        count_type count;

        friend bool
        operator<(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id + (lhs.count - 1)) < rhs.id; }

        friend bool
        operator==(gva_cache_key const& lhs, gva_cache_key const& rhs)
        { return (lhs.id == rhs.id) && (lhs.count == rhs.count); }
    }; // }}}
    
    struct gva_erase_policy
    { // {{{ gva_erase_policy implementation
        gva_erase_policy(naming::gid_type const& id, count_type count)
          : entry(id, count)
        {}

        typedef std::pair<
            gva_cache_key, boost::cache::entries::lfu_entry<gva_type>
        > entry_type;

        bool operator()(entry_type const& p) const
        { return p.first == entry; }

        gva_cache_key entry;
    }; // }}}

    typedef boost::cache::entries::lfu_entry<gva_type> gva_entry_type;

    typedef boost::cache::local_cache<
        gva_cache_key, gva_entry_type, 
        std::less<gva_entry_type>,
        boost::cache::policies::always<gva_entry_type>,
        std::map<gva_cache_key, gva_entry_type>
    > gva_cache_type;
    // }}}

    // {{{ locality cache 
    typedef boost::cache::entries::lfu_entry<naming::gid_type>
        locality_entry_type;

    typedef boost::cache::local_cache<naming::locality, locality_entry_type>
        locality_cache_type;
    // }}}

    struct bootstrap_data_type
    { // {{{
        primary_namespace_server_type primary_ns_server;
        component_namespace_server_type component_ns_server;
        symbol_namespace_server_type symbol_ns_server;
    }; // }}}

    struct hosted_data_type
    { // {{{
        primary_namespace_type primary_ns_;
        component_namespace_type component_ns_;
        symbol_namespace_type symbol_ns_;

        cache_mutex_type gva_cache_mtx_;
        gva_cache_type gva_cache_;
    
        cache_mutex_type locality_cache_mtx_;
        locality_cache_type locality_cache_;

        console_cache_type console_cache_;

        boost::atomic<std::size_t> stage;
    }; // }}}

    const runtime_mode runtime_type;
    const router_mode router_type;

    boost::shared_ptr<bootstrap_data_type> bootstrap;
    boost::shared_ptr<hosted_data_type> hosted;

    legacy_router(
        util::runtime_configuration const& ini_
      , runtime_mode runtime_type_
      , router_mode router_type_
    ):
        runtime_type(runtime_type_)
      , router_type(router_type_)
    {
        if (router_type == router_mode_bootstrap)
            start_bootstrap(ini_);
        else
            start_hosted_stage0(ini_);
    } 

    void start_bootstrap(
        util::runtime_configuration const& ini_
    ) {
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
                static_cast<void*>(&bootstrap->symbol_ns_server.get()));
        
        bootstrap->primary_ns_server.bind_locality(ep, 3);
        bootstrap->symbol_ns_server.bind("/locality(agas#0)",
            naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 

        bootstrap->primary_ns_server.bind_gid
            (primary_ns_server.get_base_gid(), primary_gva);
        bootstrap->primary_ns_server.bind_gid
            (component_ns_server.get_base_gid(), component_gva);
        bootstrap->primary_ns_server.bind_gid
            (symbol_ns_server.get_base_gid(), symbol_gva);

        if (runtime_type == runtime_mode_console)
            bootstrap->symbol_ns_server.bind("/locality(console)",
                naming::get_gid_from_prefix(HPX_AGAS_BOOTSTRAP_PREFIX)); 
    }

    void start_hosted_stage0(
        util::runtime_configuration const& ini_
    ) {
        hosted = boost::make_shared<hosted_data_type>();

        hosted->stage.store(0);

        hosted->gva_cache_.reserve(ini_.get_agas_gva_cache_size());
        hosted->locality_cache_.reserve(ini_.get_agas_locality_cache_size());
    }

    bool is_bootstrap() const
    { return router_type == router_mode_bootstrap; } 

    bool is_console() const
    { return runtime_type == runtime_mode_console; }
 
    bool is_smp_mode() const
    { return false; } 

    bool get_prefix(naming::locality const& l, naming::gid_type& prefix,
                    bool self = true, bool try_cache = true,
                    error_code& ec = throws) 
    { // {{{ get_prefix implementation
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        if (try_cache && !is_bootstrap() &&
            get_prefix_cached(l, prefix, self, ec))
            return false;

        address addr = address::from_string(l.get_address());

        endpoint_type ep(addr, l.get_port()); 

        if (self)
        {
            binding_type r;

            if (is_bootstrap())
                r = bootstrap->primary_ns_server.bind_locality(ep, 0);
            else
                r = hosted->primary_ns_.bind(ep, 0);

            prefix = at_c<2>(r);

            if (!is_bootstrap() && prefix != naming::invalid_gid)
            {
                typename cache_mutex_type::scoped_lock
                    lock(hosted->locality_cache_mtx_);
                hosted->locality_cache_.insert(l, prefix);
            }

            if (at_c<3>(r) && is_console())
            {
                if (!registerid("/locality(console)", at_c<2>(r), ec))
                {
                    HPX_THROWS_IN_CURRENT_FUNC_IF(ec, duplicate_console, 
                        "a console locality is already registered");
                }
            } 

            return at_c<3>(r);
        }
        
        else 
        {
            if (is_bootstrap())
                prefix = at_c<0>
                    (bootstrap->primary_ns_server.resolve_locality(ep));
            else
                prefix = at_c<0>(hosted->primary_ns_.resolve(ep)); 

            if (!is_bootstrap() && prefix != naming::invalid_gid)
            {
                typename cache_mutex_type::scoped_lock
                    lock(hosted->locality_cache_mtx_);
                hosted->locality_cache_.insert(l, prefix);
            }

            return false;
        }
    } // }}}
    
    bool get_prefix_cached(naming::locality const& l, naming::gid_type& prefix,
                           bool self = true, error_code& ec = throws) 
    { // {{{
        if (is_bootstrap())
            return get_prefix(l, prefix, self, false, ec);
 
        locality_entry_type e;

        typename cache_mutex_type::scoped_lock
            lock(hosted->locality_cache_mtx_);

        if (hosted->locality_cache_.get_entry(l, e))
        {
            prefix = e.get();
            return true;
        }

        return false; 
    } // }}}

    bool get_console_prefix(naming::gid_type& prefix,
                            bool try_cache = true, error_code& ec = throws) 
    { // {{{
        if (try_cache && !is_bootstrap())
        {
            if (hosted->console_cache_)
            {
                prefix = naming::get_gid_from_prefix(bootstrap->console_cache_);
                return true;
            }
        }

        if (is_bootstrap())
            prefix = bootstrap->symbol_ns_server.resolve("/locality(console)");

        else {
            prefix = hosted->symbol_ns_.resolve("/locality(console)");

            if (prefix != naming::invalid_gid)
                hosted->console_cache_.store
                    (naming::get_prefix_from_gid(prefix));
        }

        return prefix;
    } // }}}

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      component_id_type type, error_code& ec = throws) 
    { // {{{ get_prefixes implementation
        typedef typename prefixes_type::const_iterator iterator;

        if (type != components::component_invalid)
        {
            prefixes_type raw_prefixes;

            if (is_bootstrap())
                raw_prefixes = bootstrap->component_ns_server.resolve_id(type);
            else
                raw_prefixes = hosted->component_ns_.resolve(type);
    
            if (raw_prefixes.empty())
                return false;
    
            iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
            for (; it != end; ++it) 
                prefixes.push_back(naming::get_gid_from_prefix(*it));
    
            return true; 
        }

        prefixes_type raw_prefixes;

        if (is_bootstrap())
            raw_prefixes = bootstrap->primary_ns_server.localities();
        else
            raw_prefixes = hosted->primary_ns_.localities();
    
        if (raw_prefixes.empty())
            return false;
    
        iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
        for (; it != end; ++it) 
            prefixes.push_back(naming::get_gid_from_prefix(*it));
    
        return true; 
    } // }}} 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      error_code& ec = throws) 
    { return get_prefixes(prefixes, components::component_invalid, ec); }

    component_id_type
    get_component_id(std::string const& name, error_code& ec = throws) 
    { /// {{{
        if (is_bootstrap())
            return bootstrap->component_ns_server.bind_name(name);
        else
            return hosted->component_ns_.bind(name);
    } // }}} 

    component_id_type
    register_factory(naming::gid_type const& prefix, std::string const& name,
                     error_code& ec = throws) 
    { // {{{
        if (is_bootstrap())
            return bootstrap->component_ns_server.bind_prefix
                (name, naming::get_prefix_from_gid(prefix));
        else
            return hosted->component_ns_.bind
                (name, naming::get_prefix_from_gid(prefix));
    } // }}}

    bool get_id_range(naming::locality const& l, count_type count, 
                      naming::gid_type& lower_bound,
                      naming::gid_type& upper_bound, error_code& ec = throws) 
    { // {{{ get_id_range implementation
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(l.get_address());

        endpoint_type ep(addr, l.get_port()); 
         
        binding_type range;

        if (is_bootstrap())
            range = bootstrap->primary_ns_server.bind_locality(ep, count);
        else
            range = hosted->primary_ns_.bind(ep, count);

        lower_bound = at_c<0>(range);
        upper_bound = at_c<1>(range);

        return lower_bound && upper_bound;
    } // }}}

    bool bind(naming::gid_type const& id, naming::address const& addr,
              error_code& ec = throws) 
    { return bind_range(id, 1, addr, 0, ec); }

    bool bind_range(naming::gid_type const& lower_id, count_type count, 
                    naming::address const& baseaddr, offset_type offset,
                    error_code& ec = throws) 
    { // {{{ bind_range implementation
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(baseaddr.locality_.get_address());

        endpoint_type ep(addr, baseaddr.locality_.get_port()); 
       
        // Create a global virtual address from the legacy calling convention
        // parameters.
        gva_type gva(ep, baseaddr.type_, count, baseaddr.address_, offset);
        
        if (is_bootstrap())
        {
            if (bootstrap->primary_ns_server.bind_gid(lower_id, gva)) 
                return true;
        }

        else if (hosted->primary_ns_.bind(lower_id, gva)) 
        { 
            typename cache_mutex_type::scoped_lock
                lock(hosted->gva_cache_mtx_);
            gva_cache_key key(lower_id, count);
            hosted->gva_cache_.insert(key, gva);
            return true;
        }

        return false; 
    } // }}}

    count_type
    incref(naming::gid_type const& id, count_type credits = 1,
           error_code& ec = throws) 
    {
        if (is_bootstrap())
            return bootstrap->primary_ns_server.increment(id, credits);
        else
            return hosted->primary_ns_.increment(id, credits);
    } 

    count_type
    decref(naming::gid_type const& id, component_id_type& t,
           count_type credits = 1, error_code& ec = throws) 
    { // {{{ decref implementation
        using boost::fusion::at_c;

        decrement_type r;
        
        if (is_bootstrap())
            r = bootstrap->primary_ns_server.decrement(id, credits);
        else
            r = hosted->primary_ns_.decrement(id, credits);

        if (at_c<0>(r) == 0)
            t = at_c<1>(r);

        return at_c<0>(r);
    } // }}}

    bool unbind(naming::gid_type const& id, error_code& ec = throws) 
    { return unbind_range(id, 1, ec); } 
        
    bool unbind(naming::gid_type const& id, naming::address& addr,
                error_code& ec = throws) 
    { return unbind_range(id, 1, addr, ec); }

    bool unbind_range(naming::gid_type const& lower_id, count_type count,
                      error_code& ec = throws) 
    {
        naming::address addr; 
        return unbind_range(lower_id, count, addr, ec);
    } 

    bool unbind_range(naming::gid_type const& lower_id, count_type count, 
                      naming::address& addr, error_code& ec = throws) 
    { // {{{ unbind_range implementation
        unbinding_type r;

        if (is_bootstrap())
            r = bootstrap->primary_ns_server.unbind(lower_id, count);
        else
            r = hosted->primary_ns_.unbind(lower_id, count);

        if (!is_bootstrap() && r)
        {
            typename cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            gva_erase_policy ep(lower_id, count);
            hosted->gva_cache_.erase(ep);
            addr.locality_ = r->endpoint;
            addr.type_ = r->type;
            addr.address_ = r->lva();
        }

        return r; 
    } // }}}

    bool resolve(naming::gid_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { // {{{ resolve implementation
        if (try_cache && !is_bootstrap() && resolve_cached(id, addr, ec))
            return true;
        
        gva_type gva; 

        if (is_bootstrap())
            gva = bootstrap->primary_ns_server.resolve_gid(id);
        else
            gva = hosted->primary_ns_.resolve(id);

        addr.locality_ = gva.endpoint;
        addr.type_ = gva.type;
        addr.address_ = gva.lva();

        if (HPX_LIKELY((gva.endpoint != endpoint_type()) &&
                       (gva.type != components::component_invalid) &&
                       (gva.lva() != 0)))
        {     
            if (is_bootstrap())
                return true;

            // We only insert the entry into the cache if it's valid
            typename cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
            gva_cache_key key(id);
            hosted->gva_cache_.insert(key, gva);
            return true;
        }

        return false;
    } // }}}

    bool resolve(naming::id_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { return resolve(id.get_gid(), addr, try_cache, ec); }

    bool resolve_cached(naming::gid_type const& id, naming::address& addr,
                        error_code& ec = throws) 
    { // {{{ resolve_cached implementation
        if (is_bootstrap())
            return resolve(id, addr, false, ec);

        // first look up the requested item in the cache
        gva_cache_key k(id);
        typename cache_mutex_type::scoped_lock lock(hosted->gva_cache_mtx_);
        gva_cache_key idbase;
        typename gva_cache_type::entry_type e;

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

    bool registerid(std::string const& name, naming::gid_type const& id,
                    error_code& ec = throws) 
    {
        naming::gid_type r;

        if (is_bootstrap())
            r = bootstrap->symbol_ns_server.rebind(name, id);
        else
            r = hosted->symbol_ns_.rebind(name, id);

        return r == id;
    }

    bool unregisterid(std::string const& name, error_code& ec = throws) 
    {
        if (is_bootstrap())
            return bootstrap->symbol_ns_server.unbind(name);
        else  
            return hosted->symbol_ns_.unbind(name);
    }

    bool queryid(std::string const& ns_name, naming::gid_type& id,
                 error_code& ec = throws) 
    {
        if (is_bootstrap())
            id = bootstrap->symbol_ns_server.resolve(ns_name);
        else
            id = hosted->symbol_ns_.resolve(ns_name);
        return id;         
    }
};

}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

