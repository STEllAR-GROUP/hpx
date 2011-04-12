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
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/server/symbol.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database>
struct symbol_namespace
  : components::stubs::stub_base<server::symbol_namespace<Database> >
{
    // {{{ nested types
    typedef server::symbol_namespace<Database> server_type;

    typedef typename server_type::symbol_type symbol_type;
    // }}}

    // {{{ bind dispatch
    static lcos::future_value<void>
    bind_async(naming::id_type const& gid, symbol_type const& key,
               naming::gid_type const& value)
    {
        typedef typename server_type::bind_action action_type;
        return lcos::eager_future<action_type, void>(gid, key, value);
    }

    static void bind(naming::id_type const& gid, symbol_type const& key,
                     naming::gid_type const& value)
    { return bind_async(gid, key, value).get(); } 
    // }}}

    // {{{ resolve dispatch 
    static lcos::future_value<naming::gid_type>
    resolve_async(naming::id_type const& gid, symbol_type const& key)
    {
        typedef typename server_type::resolve_action action_type;
        return lcos::eager_future<action_type, naming::gid_type>(gid, key);
    }
    
    static naming::gid_type
    resolve(naming::id_type const& gid, symbol_type const& key)
    { return resolve_async(gid, key).get(); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<void>
    unbind_async(naming::id_type const& gid, symbol_type const& key)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, void>(gid, key);
    }
    
    static void unbind(naming::id_type const& gid, symbol_type const& key)
    { return unbind_async(gid, key).get(); } 
    // }}}
};            

}}}

#endif // HPX_28443929_CB68_43ED_B134_F60602A344DD

