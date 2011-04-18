////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7)
#define HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/stubs/symbol.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct symbol_namespace
  : components::client_base<
      symbol_namespace<Database>,
      stubs::symbol_namespace<Database>
    >
{
    // {{{ nested types 
    typedef components::client_base<
        symbol_namespace<Database>,
        stubs::symbol_namespace<Database>
    > base_type;

    typedef server::symbol_namespace<Database> server_type;

    typedef typename server_type::symbol_type symbol_type;
    // }}}

    explicit symbol_namespace(naming::id_type const& id = naming::invalid_id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::future_value<bool>
    bind_async(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::bind_async(this->gid_, key, gid); }

    bool bind(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::bind(this->gid_, key, gid); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve interface 
    lcos::future_value<naming::gid_type> resolve_async(symbol_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    naming::gid_type resolve(symbol_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::future_value<void> unbind_async(symbol_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    void unbind(symbol_type const& key)
    { return this->base_type::unbind(this->gid_, key); }
};            

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

