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
#include <hpx/runtime/agas/namespace/server/component.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol>
struct component_namespace 
{
    // {{{ nested types
    typedef server::component_namespace<Database, Protocol> server_type; 

    typedef typename server_type::response_type response_type;
    typedef typename server_type::component_name_type component_name_type;
    typedef typename server_type::component_id_type component_id_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    // }}}

    // {{{ bind_prefix dispatch
    static lcos::future_value<response_type>
    bind_prefix_async(naming::id_type const& gid, component_name_type const& key,
                      prefix_type prefix)
    {
        typedef typename server_type::bind_prefix_action action_type;
        return lcos::eager_future<action_type, response_type>
            (gid, key, prefix);
    }

    static response_type
    bind_prefix(naming::id_type const& gid, component_name_type const& key,
                prefix_type prefix, error_code& ec = throws)
    {
        return bind_prefix_async(gid, key, prefix).get(ec);
    } 
    // }}}
    
    // {{{ bind_name dispatch 
    static lcos::future_value<response_type>
    bind_name_async(naming::id_type const& gid, component_name_type const& key)
    {
        typedef typename server_type::bind_name_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    bind_name(naming::id_type const& gid, component_name_type const& key,
              error_code& ec = throws)
    { return bind_name_async(gid, key).get(ec); } 
    // }}}

    // {{{ resolve_id dispatch
    static lcos::future_value<response_type>
    resolve_id_async(naming::id_type const& gid, component_id_type key)
    {
        typedef typename server_type::resolve_id_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    resolve_id(naming::id_type const& gid, component_id_type key,
               error_code& ec = throws)
    { return resolve_id_async(gid, key).get(ec); } 
    // }}}

    // {{{ resolve_name dispatch 
    static lcos::future_value<response_type>
    resolve_name_async(naming::id_type const& gid,
                       component_name_type const& key)
    {
        typedef typename server_type::resolve_name_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    resolve_name(naming::id_type const& gid, component_name_type const& key,
                 error_code& ec = throws)
    { return resolve_name_async(gid, key).get(ec); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<response_type>
    unbind_async(naming::id_type const& gid, component_name_type const& key,
                 error_code& ec = throws)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    unbind(naming::id_type const& gid, component_name_type const& key,
           error_code& ec = throws)
    { return unbind_async(gid, key).get(ec); } 
    // }}}
};            

}}}

#endif // HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

