////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7)
#define HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/symbol.hpp>

namespace hpx { namespace agas 
{

// TODO: add error code parameters
template <typename Database, typename Protocol>
struct symbol_namespace :
    components::client_base<
        symbol_namespace<Database, Protocol>,
        stubs::symbol_namespace<Database, Protocol>
    >
{
    // {{{ nested types 
    typedef components::client_base<
        symbol_namespace<Database, Protocol>,
        stubs::symbol_namespace<Database, Protocol>
    > base_type; 

    typedef server::symbol_namespace<Database, Protocol> server_type;

    typedef typename server_type::response_type response_type;
    typedef typename server_type::iterate_function_type iterate_function_type;
    typedef typename server_type::symbol_type symbol_type;
    // }}}

    explicit symbol_namespace(naming::id_type const& id =
      naming::id_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB,
                      naming::id_type::unmanaged))
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::promise<response_type>
    bind_async(symbol_type const& key, naming::gid_type const& gid)
    { return this->base_type::bind_async(this->gid_, key, gid); }

    response_type bind(symbol_type const& key, naming::gid_type const& gid,
                       error_code& ec = throws)
    { return this->base_type::bind(this->gid_, key, gid, ec); }
    
    ///////////////////////////////////////////////////////////////////////////
    // resolve interface 
    lcos::promise<response_type> resolve_async(symbol_type const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    response_type resolve(symbol_type const& key, error_code& ec = throws)
    { return this->base_type::resolve(this->gid_, key, ec); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::promise<response_type> unbind_async(symbol_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    response_type unbind(symbol_type const& key, error_code& ec = throws)
    { return this->base_type::unbind(this->gid_, key, ec); }

    ///////////////////////////////////////////////////////////////////////////
    // iterate interface 
    lcos::promise<response_type>
    iterate_async(iterate_function_type const& f)
    { return this->base_type::iterate_async(this->gid_, f); }
    
    response_type iterate(iterate_function_type const& f,
                          error_code& ec = throws)
    { return this->base_type::iterate(this->gid_, f, ec); }
};            

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

