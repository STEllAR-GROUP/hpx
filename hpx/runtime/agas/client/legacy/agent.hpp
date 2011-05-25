////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_15D904C7_CD18_46E1_A54A_65059966A34F)
#define HPX_15D904C7_CD18_46E1_A54A_65059966A34F

#include <boost/noncopyable.hpp>

#include <hpx/runtime/agas/client/legacy/user.hpp>
#include <hpx/runtime/agas/client/legacy/bootstrap.hpp>

namespace hpx { namespace agas { namespace legacy
{

// TODO: pass error codes once they're implemented in AGAS.
template <typename Base>
struct agent_base : Base
{
    // {{{ types 
    typedef Base base_type;

    typedef typename base_type::primary_namespace_type
        primary_namespace_type;

    typedef typename base_type::component_namespace_type
        component_namespace_type;

    typedef typename base_type::symbol_namespace_type
        symbol_namespace_type;

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
    
    typedef typename base_type::cache_mutex_type cache_mutex_type;
    typedef typename base_type::erase_policy erase_policy;
    typedef typename base_type::cache_key cache_key;
    typedef typename base_type::cache_type cache_type;
    // }}}

    const runtime_mode mode;

    agent_base(util::runtime_configuration const& ini_, runtime_mode mode_)
        : base_type(ini_, mode_), mode(mode_) {}

    bool is_console() const
    { return mode == runtime_mode_console; }
    
    bool is_smp_mode() const
    { return false; } 

    bool get_prefix(naming::locality const& l, naming::gid_type& prefix,
                    bool self = true, error_code& ec = throws) 
    { // {{{ get_prefix implementation
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(l.get_address());

        endpoint_type ep(addr, l.get_port()); 

        if (self)
        {
            binding_type r = this->primary_ns_.bind(ep, 0);
            prefix = at_c<2>(r);
            return at_c<3>(r);
        }
        
        else 
        {
            prefix = at_c<0>(this->primary_ns_.resolve(ep)); 
            return false;
        }
    } // }}}

    bool get_console_prefix(naming::gid_type& prefix, error_code& ec = throws) 
    {
        prefix = this->symbol_ns_.resolve("/locality(console)");
        return prefix;
    } 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      component_id_type type, error_code& ec = throws) 
    { // {{{ get_prefixes implementation
        typedef typename prefixes_type::const_iterator iterator;

        if (type != components::component_invalid)
        {
            prefixes_type raw_prefixes = this->component_ns_.resolve(type);
    
            if (raw_prefixes.empty())
                return false;
    
            iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
            for (; it != end; ++it) 
                prefixes.push_back(naming::get_gid_from_prefix(*it));
    
            return true; 
        }

        prefixes_type raw_prefixes = this->primary_ns_.localities();
    
        if (raw_prefixes.empty())
            return false;
    
        iterator it = raw_prefixes.begin(), end = raw_prefixes.end();
    
        for (; it != end; ++it) 
            prefixes.push_back(naming::get_gid_from_prefix(*it));
    
        return true; 
    } // }}} 

    bool get_prefixes(std::vector<naming::gid_type>& prefixes,
                      error_code& ec = throws) 
    { return get_prefixes(prefixes, components::component_invalid); }

    component_id_type
    get_component_id(std::string const& name, error_code& ec = throws) 
    { return this->component_ns_.bind(name); } 

    component_id_type
    register_factory(naming::gid_type const& prefix, std::string const& name,
                     error_code& ec = throws) 
    {
        return this->component_ns_.bind
            (name, naming::get_prefix_from_gid(prefix));
    } 

    bool get_id_range(naming::locality const& l, count_type count, 
                      naming::gid_type& lower_bound,
                      naming::gid_type& upper_bound, error_code& ec = throws) 
    { // {{{ get_id_range implementation
        using boost::asio::ip::address;
        using boost::fusion::at_c;

        address addr = address::from_string(l.get_address());

        endpoint_type ep(addr, l.get_port()); 
         
        binding_type range = this->primary_ns_.bind(ep, count);

        lower_bound = at_c<0>(range);
        upper_bound = at_c<1>(range);

        return lower_bound && upper_bound;
    } // }}}

    bool bind(naming::gid_type const& id, naming::address const& addr,
              error_code& ec = throws) 
    { return bind_range(id, 1, addr, 0); }

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
        
        if (this->primary_ns_.bind(lower_id, gva)) 
        { 
            typename cache_mutex_type::scoped_lock lock(this->cache_mtx_);
            cache_key key(lower_id, count);
            this->gva_cache_.insert(key, gva);

            return true;
        }

        return false; 
    } // }}}

    count_type
    incref(naming::gid_type const& id, count_type credits = 1,
           error_code& ec = throws) 
    { return this->primary_ns_.increment(id, credits); } 

    count_type
    decref(naming::gid_type const& id, component_id_type& t,
           count_type credits = 1, error_code& ec = throws) 
    { // {{{ decref implementation
        using boost::fusion::at_c;

        decrement_type r = this->primary_ns_.decrement(id, credits);

        if (at_c<0>(r) == 0)
            t = at_c<1>(r);

        return at_c<0>(r);
    } // }}}

    bool unbind(naming::gid_type const& id, error_code& ec = throws) 
    { return unbind_range(id, 1); } 
        
    bool unbind(naming::gid_type const& id, naming::address& addr,
                error_code& ec = throws) 
    { return unbind_range(id, 1, addr); }

    bool unbind_range(naming::gid_type const& lower_id, count_type count,
                      error_code& ec = throws) 
    {
        naming::address addr; 
        return unbind_range(lower_id, count, addr);
    } 

    bool unbind_range(naming::gid_type const& lower_id, count_type count, 
                      naming::address& addr, error_code& ec = throws) 
    { // {{{ unbind_range implementation
        unbinding_type r = this->primary_ns_.unbind(lower_id, count);

        if (r)
        {
            typename cache_mutex_type::scoped_lock lock(this->cache_mtx_);
            erase_policy ep(lower_id, count);
            this->gva_cache_.erase(ep);
            addr.locality_ = r->endpoint;
            addr.type_ = r->type;
            addr.address_ = r->lva();
        }

        return r; 
    } // }}}

    bool resolve(naming::gid_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { // {{{ resolve implementation
        if (try_cache && resolve_cached(id, addr))
            return true;
        
        gva_type gva = this->primary_ns_.resolve(id);

        addr.locality_ = gva.endpoint;
        addr.type_ = gva.type;
        addr.address_ = gva.lva();

        if (HPX_LIKELY((gva.endpoint != endpoint_type()) &&
                       (gva.type != components::component_invalid) &&
                       (gva.lva() != 0)))
        {     
            // We only insert the entry into the cache if it's valid
            typename cache_mutex_type::scoped_lock lock(this->cache_mtx_);
            cache_key key(id);
            this->gva_cache_.insert(key, gva);
            return true;
        }

        return false;
    } // }}}

    bool resolve(naming::id_type const& id, naming::address& addr,
                 bool try_cache = true, error_code& ec = throws) 
    { return resolve(id.get_gid(), addr, try_cache); }

    bool resolve_cached(naming::gid_type const& id, naming::address& addr,
                        error_code& ec = throws) 
    { // {{{ resolve_cached implementation
        // first look up the requested item in the cache
        cache_key k(id);
        typename cache_mutex_type::scoped_lock lock(this->cache_mtx_);
        cache_key idbase;
        typename cache_type::entry_type e;

        // Check if the entry is currently in the cache
        if (this->gva_cache_.get_entry(k, idbase, e))
        {
            if (HPX_UNLIKELY(id.get_msb() != idbase.id.get_msb()))
            {
                HPX_THROW_IN_CURRENT_FUNC(bad_parameter, 
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
        naming::gid_type r = this->symbol_ns_.rebind(name, id);
        return r == id;
    }

    bool unregisterid(std::string const& name, error_code& ec = throws) 
    { return this->symbol_ns_.unbind(name); }

    bool queryid(std::string const& ns_name, naming::gid_type& id,
                 error_code& ec = throws) 
    {
        id = this->symbol_ns_.resolve(ns_name);
        return id;         
    }
};

template <typename Database>
struct user_agent : agent_base<user<Database> >, boost::noncopyable
{
    typedef agent_base<user<Database> > base_type;

    user_agent(util::runtime_configuration const& ini_
                  = util::runtime_configuration(), 
               runtime_mode mode = runtime_mode_worker)
        : base_type(ini_, mode) {} 
};

template <typename Database>
struct bootstrap_agent : agent_base<bootstrap<Database> >, boost::noncopyable
{
    typedef agent_base<bootstrap<Database> > base_type;

    bootstrap_agent(util::runtime_configuration const& ini_
                      = util::runtime_configuration(), 
                    runtime_mode mode = runtime_mode_worker)
        : base_type(ini_, mode) {} 
};

}}}

#endif // HPX_15D904C7_CD18_46E1_A54A_65059966A34F

