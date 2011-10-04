////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7)
#define HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/symbol_namespace.hpp>

namespace hpx { namespace agas 
{

struct symbol_namespace :
    components::client_base<symbol_namespace, stubs::symbol_namespace>
{
    // {{{ nested types 
    typedef components::client_base<
        symbol_namespace, stubs::symbol_namespace
    > base_type; 

    typedef server::symbol_namespace server_type;

    typedef server_type::iterate_function_type iterate_function_type;
    // }}}

    explicit symbol_namespace(naming::id_type const& id =
      naming::id_type(HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB,
                      naming::id_type::unmanaged))
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::promise<response>
    bind_async(std::string const& key, naming::gid_type const& gid)
    { return this->base_type::bind_async(this->gid_, key, gid); }

    response bind(std::string const& key, naming::gid_type const& gid,
                       error_code& ec = throws)
    { return this->base_type::bind(this->gid_, key, gid, ec); }
    
    ///////////////////////////////////////////////////////////////////////////
    // resolve interface 
    lcos::promise<response> resolve_async(std::string const& key)
    { return this->base_type::resolve_async(this->gid_, key); }
    
    response resolve(std::string const& key, error_code& ec = throws)
    { return this->base_type::resolve(this->gid_, key, ec); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::promise<response> unbind_async(std::string const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    response unbind(std::string const& key, error_code& ec = throws)
    { return this->base_type::unbind(this->gid_, key, ec); }

    ///////////////////////////////////////////////////////////////////////////
    // iterate interface 
    lcos::promise<response>
    iterate_async(iterate_function_type const& f)
    { return this->base_type::iterate_async(this->gid_, f); }
    
    response iterate(iterate_function_type const& f,
                          error_code& ec = throws)
    { return this->base_type::iterate(this->gid_, f, ec); }
};            

}}

#endif // HPX_2A00BD90_B331_44BC_AF02_06787ABC50E7

