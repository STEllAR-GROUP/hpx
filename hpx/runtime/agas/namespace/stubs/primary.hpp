////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F)
#define HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/server/primary.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Server>
struct primary_namespace_base : components::stubs::stub_base<Server>
{
    // {{{ nested types
    typedef components::stubs::stub_base<Server> base_type;
    typedef Server server_type; 

    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::offset_type offset_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    typedef typename server_type::binding_type binding_type;
    typedef typename server_type::unbinding_type unbinding_type;
    typedef typename server_type::locality_type locality_type;
    typedef typename server_type::decrement_type decrement_type;
    // }}}

    // {{{ bind_locality dispatch
    static lcos::future_value<binding_type>
    bind_locality_async(naming::id_type const& gid, endpoint_type const& ep,
                        count_type count)
    {
        typedef typename server_type::bind_locality_action action_type;
        return lcos::eager_future<action_type, binding_type>(gid, ep, count);
    }

    static binding_type
    bind_locality(naming::id_type const& gid, endpoint_type const& ep,
                  count_type count)
    { return bind_locality_async(gid, ep, count).get(); } 
    // }}}
    
    // {{{ bind_gid dispatch
    static lcos::future_value<bool>
    bind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                   gva_type const& gva)
    {
        typedef typename server_type::bind_gid_action action_type;
        return lcos::eager_future<action_type, bool>(gid, id, gva);
    }

    static bool
    bind_gid(naming::id_type const& gid, naming::gid_type const& id,
             gva_type const& gva)
    { return bind_gid_async(gid, id, gva).get(); } 
    // }}}

    // {{{ resolve_locality dispatch
    static lcos::future_value<locality_type>
    resolve_locality_async(naming::id_type const& gid, endpoint_type const& ep)
    {
        typedef typename server_type::resolve_locality_action action_type;
        return lcos::eager_future<action_type, locality_type>(gid, ep);
    }
    
    static locality_type
    resolve_locality(naming::id_type const& gid, endpoint_type const& ep)
    { return resolve_locality_async(gid, ep).get(); } 
    // }}}

    // {{{ resolve_gid dispatch 
    static lcos::future_value<gva_type>
    resolve_gid_async(naming::id_type const& gid, naming::gid_type const& key)
    {
        typedef typename server_type::resolve_gid_action action_type;
        return lcos::eager_future<action_type, gva_type>(gid, key);
    }
    
    static gva_type
    resolve_gid(naming::id_type const& gid, naming::gid_type const& key)
    { return resolve_gid_async(gid, key).get(); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<unbinding_type>
    unbind_async(naming::id_type const& gid, naming::gid_type const& id,
                 count_type count)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, unbinding_type>(gid, id, count);
    }
    
    static unbinding_type
    unbind(naming::id_type const& gid, naming::gid_type const& id,
           count_type count)
    { return unbind_async(gid, id, count).get(); } 
    // }}}
    
    // {{{ increment dispatch 
    static lcos::future_value<count_type>
    increment_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::increment_action action_type;
        return lcos::eager_future<action_type, count_type>(gid, key, count);
    }
    
    static count_type
    increment(naming::id_type const& gid, naming::gid_type const& key,
              count_type count)
    { return increment_async(gid, key, count).get(); } 
    // }}}
    
    // {{{ decrement dispatch 
    static lcos::future_value<decrement_type>
    decrement_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::decrement_action action_type;
        return lcos::eager_future<action_type, decrement_type>
            (gid, key, count);
    }
    
    static decrement_type
    decrement(naming::id_type const& gid, naming::gid_type const& key,
              count_type count)
    { return decrement_async(gid, key, count).get(); } 
    // }}}
    
    // {{{ localities dispatch 
    static lcos::future_value<prefixes_type>
    localities_async(naming::id_type const& gid)
    {
        typedef typename server_type::localities_action action_type;
        return lcos::eager_future<action_type, prefixes_type>(gid);
    }
    
    static prefixes_type
    localities(naming::id_type const& gid)
    { return localities_async(gid).get(); } 
    // }}}
};            

template <typename Database, typename Protocol> 
struct primary_namespace : primary_namespace_base<
    server::primary_namespace<Database, Protocol>
> { };

template <typename Database, typename Protocol> 
struct bootstrap_primary_namespace : primary_namespace_base<
    server::bootstrap_primary_namespace<Database, Protocol>
> { };

}}}

#endif // HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

