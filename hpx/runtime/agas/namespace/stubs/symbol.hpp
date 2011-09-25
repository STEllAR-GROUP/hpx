////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_28443929_CB68_43ED_B134_F60602A344DD)
#define HPX_28443929_CB68_43ED_B134_F60602A344DD

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/agas/namespace/server/symbol.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol>
struct symbol_namespace 
{
    // {{{ nested types
    typedef server::symbol_namespace<Database, Protocol> server_type; 

    typedef typename server_type::response_type response_type;
    typedef typename server_type::iterate_function_type iterate_function_type;
    typedef typename server_type::symbol_type symbol_type;
    // }}}

    // {{{ bind dispatch
    static lcos::future_value<response_type>
    bind_async(naming::id_type const& gid, symbol_type const& key,
               naming::gid_type const& value)
    {
        typedef typename server_type::bind_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key, value);
    }

    static response_type bind(naming::id_type const& gid, symbol_type const& key,
                              naming::gid_type const& value,
                              error_code& ec = throws)
    { return bind_async(gid, key, value).get(ec); } 
    // }}}
    
    // {{{ resolve dispatch 
    static lcos::future_value<response_type>
    resolve_async(naming::id_type const& gid, symbol_type const& key)
    {
        typedef typename server_type::resolve_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    resolve(naming::id_type const& gid, symbol_type const& key,
            error_code& ec = throws)
    { return resolve_async(gid, key).get(ec); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<response_type>
    unbind_async(naming::id_type const& gid, symbol_type const& key)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    unbind(naming::id_type const& gid, symbol_type const& key,
           error_code& ec = throws)
    { return unbind_async(gid, key).get(ec); } 
    // }}}

    // {{{ iterate dispatch 
    static lcos::future_value<response_type>
    iterate_async(naming::id_type const& gid, iterate_function_type const& f)
    {
        typedef typename server_type::iterate_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, f);
    }
    
    static response_type
    iterate(naming::id_type const& gid, iterate_function_type const& f,
            error_code& ec = throws)
    { return iterate_async(gid, f).get(ec); } 
    // }}}
};            

}}}

#endif // HPX_28443929_CB68_43ED_B134_F60602A344DD

