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

// TODO: add error code parameters
template <typename Base, typename Server>
struct symbol_namespace_base : Base
{
    // {{{ nested types 
    typedef Base base_type; 
    typedef Server server_type;

    typedef typename server_type::symbol_type symbol_type;
    // }}}

    explicit symbol_namespace_base(naming::id_type const& id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::future_value<bool>
    bind_async(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::bind_async(this->gid_, key, gid); }

    bool bind(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::bind(this->gid_, key, gid); }
    
    ///////////////////////////////////////////////////////////////////////////
    // rebind interface 
    lcos::future_value<naming::gid_type>
    rebind_async(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::rebind_async(this->gid_, key, gid); }

    naming::gid_type rebind(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::rebind(this->gid_, key, gid); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve interface 
    lcos::future_value<naming::gid_type> resolve_async(symbol_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    naming::gid_type resolve(symbol_type const& key)
    { return this->base_type::resolve(this->gid_, key); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::future_value<bool> unbind_async(symbol_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    bool unbind(symbol_type const& key)
    { return this->base_type::unbind(this->gid_, key); }
};            

template <typename Database>
struct symbol_namespace : symbol_namespace_base<
    components::client_base<
        symbol_namespace<Database>,
        stubs::symbol_namespace<Database>
    >,
    server::symbol_namespace<Database>
> {
    typedef symbol_namespace_base< 
        components::client_base<
            symbol_namespace<Database>,
            stubs::symbol_namespace<Database>
        >,
        server::symbol_namespace<Database>
    > base_type;

    explicit symbol_namespace(naming::id_type const& id = naming::invalid_id)
        : base_type(id) {}
};

template <typename Database>
struct bootstrap_symbol_namespace : symbol_namespace_base<
    components::client_base<
        bootstrap_symbol_namespace<Database>,
        stubs::bootstrap_symbol_namespace<Database>
    >,
    server::bootstrap_symbol_namespace<Database>
> {
    typedef symbol_namespace_base< 
        components::client_base<
            bootstrap_symbol_namespace<Database>,
            stubs::bootstrap_symbol_namespace<Database>
        >,
        server::bootstrap_symbol_namespace<Database>
    > base_type;

    explicit bootstrap_symbol_namespace(naming::id_type const& id
                                             = naming::invalid_id)
        : base_type(id) {}
};

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

