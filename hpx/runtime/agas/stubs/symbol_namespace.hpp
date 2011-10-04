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
#include <hpx/runtime/agas/server/symbol_namespace.hpp>

namespace hpx { namespace agas { namespace stubs
{

struct symbol_namespace 
{
    // {{{ nested types
    typedef server::symbol_namespace server_type; 

    typedef server_type::iterate_symbols_function_type
        iterate_symbols_function_type;
    // }}}

    // {{{ bind dispatch
    static lcos::promise<response>
    bind_async(naming::id_type const& gid, std::string const& key,
               naming::gid_type const& value)
    {
        typedef server_type::bind_action action_type;
        return lcos::eager_future<action_type, response>(gid, key, value);
    }

    static response bind(naming::id_type const& gid, std::string const& key,
                              naming::gid_type const& value,
                              error_code& ec = throws)
    { return bind_async(gid, key, value).get(ec); } 
    // }}}
    
    // {{{ resolve dispatch 
    static lcos::promise<response>
    resolve_async(naming::id_type const& gid, std::string const& key)
    {
        typedef server_type::resolve_action action_type;
        return lcos::eager_future<action_type, response>(gid, key);
    }
    
    static response
    resolve(naming::id_type const& gid, std::string const& key,
            error_code& ec = throws)
    { return resolve_async(gid, key).get(ec); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::promise<response>
    unbind_async(naming::id_type const& gid, std::string const& key)
    {
        typedef server_type::unbind_action action_type;
        return lcos::eager_future<action_type, response>(gid, key);
    }
    
    static response
    unbind(naming::id_type const& gid, std::string const& key,
           error_code& ec = throws)
    { return unbind_async(gid, key).get(ec); } 
    // }}}

    // {{{ iterate dispatch 
    static lcos::promise<response>
    iterate_async(naming::id_type const& gid,
                  iterate_symbols_function_type const& f)
    {
        typedef server_type::iterate_action action_type;
        return lcos::eager_future<action_type, response>(gid, f);
    }
    
    static response
    iterate(naming::id_type const& gid, iterate_symbols_function_type const& f,
            error_code& ec = throws)
    { return iterate_async(gid, f).get(ec); } 
    // }}}
};            

}}}

#endif // HPX_28443929_CB68_43ED_B134_F60602A344DD

