////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D)
#define HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Base, typename Server>
struct component_namespace_base : Base 
{
    // {{{ nested types
/*    typedef components::stubs::stub_base<Server> base_type; */
    typedef Server server_type; 

    typedef typename server_type::component_name_type component_name_type;
    typedef typename server_type::component_id_type component_id_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    // }}}

    // {{{ bind_prefix dispatch
    static lcos::future_value<component_id_type>
    bind_prefix_async(naming::id_type const& gid, component_name_type const& key,
                      prefix_type prefix)
    {
        typedef typename server_type::bind_prefix_action action_type;
        return lcos::eager_future<action_type, component_id_type>
            (gid, key, prefix);
    }

    static component_id_type
    bind_prefix(naming::id_type const& gid, component_name_type const& key,
                prefix_type prefix)
    { return bind_prefix_async(gid, key, prefix).get(); } 
    // }}}
    
    // {{{ bind_name dispatch 
    static lcos::future_value<component_id_type>
    bind_name_async(naming::id_type const& gid, component_name_type const& key)
    {
        typedef typename server_type::bind_name_action action_type;
        return lcos::eager_future<action_type, component_id_type>(gid, key);
    }
    
    static component_id_type
    bind_name(naming::id_type const& gid, component_name_type const& key)
    { return bind_name_async(gid, key).get(); } 
    // }}}

    // {{{ resolve_id dispatch
    static lcos::future_value<prefixes_type>
    resolve_id_async(naming::id_type const& gid, component_id_type key)
    {
        typedef typename server_type::resolve_id_action action_type;
        return lcos::eager_future<action_type, prefixes_type>(gid, key);
    }
    
    static prefixes_type
    resolve_id(naming::id_type const& gid, component_id_type key)
    { return resolve_id_async(gid, key).get(); } 
    // }}}

    // {{{ resolve_name dispatch 
    static lcos::future_value<component_id_type>
    resolve_name_async(naming::id_type const& gid,
                       component_name_type const& key)
    {
        typedef typename server_type::resolve_name_action action_type;
        return lcos::eager_future<action_type, component_id_type>(gid, key);
    }
    
    static component_id_type
    resolve_name(naming::id_type const& gid, component_name_type const& key)
    { return resolve_name_async(gid, key).get(); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<bool>
    unbind_async(naming::id_type const& gid, component_name_type const& key)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, bool>(gid, key);
    }
    
    static bool
    unbind(naming::id_type const& gid, component_name_type const& key)
    { return unbind_async(gid, key).get(); } 
    // }}}
};            

}}}

#endif // HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

