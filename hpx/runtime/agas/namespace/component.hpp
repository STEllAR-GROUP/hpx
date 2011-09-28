////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6)
#define HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/component.hpp>

namespace hpx { namespace agas 
{

// TODO: add error code parameters
template <typename Database, typename Protocol>
struct component_namespace :
    components::client_base<
        component_namespace<Database, Protocol>,
        stubs::component_namespace<Database, Protocol>
    >
{
    // {{{ nested types 
    typedef components::client_base<
        component_namespace<Database, Protocol>,
        stubs::component_namespace<Database, Protocol>
    > base_type; 

    typedef server::component_namespace<Database, Protocol> server_type;
    
    typedef typename server_type::response_type response_type;
    typedef typename server_type::component_name_type component_name_type;
    typedef typename server_type::component_id_type component_id_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    // }}}

    explicit component_namespace(naming::id_type const& id =
      naming::id_type(HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB,
                      naming::id_type::unmanaged))
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::promise<response_type>
    bind_async(component_name_type const& key, prefix_type prefix)
    { return this->base_type::bind_prefix_async(this->gid_, key, prefix); }

    response_type bind(component_name_type const& key, prefix_type prefix,
                       error_code& ec = throws)
    { return this->base_type::bind_prefix(this->gid_, key, prefix, ec); }

    lcos::promise<response_type>
    bind_async(component_name_type const& key, naming::gid_type const& prefix)
    {
        return this->base_type::bind_prefix_async
            (this->gid_, key, naming::get_prefix_from_gid(prefix));
    }

    response_type bind(component_name_type const& key,
                       naming::gid_type const& prefix,
                       error_code& ec = throws)
    {
        return this->base_type::bind_prefix
            (this->gid_, key, naming::get_prefix_from_gid(prefix), ec);
    }
    
    lcos::promise<response_type> bind_async(component_name_type const& key)
    { return this->base_type::bind_name_async(this->gid_, key); }
    
    response_type bind(component_name_type const& key, error_code& ec = throws)
    { return this->base_type::bind_name(this->gid_, key, ec); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve_id and resolve_name interface 
    lcos::promise<response_type> resolve_async(component_id_type key)
    { return this->base_type::resolve_id_async(this->gid_, key); }
    
    response_type resolve(component_id_type key, error_code& ec = throws)
    { return this->base_type::resolve_id(this->gid_, key, ec); }

    lcos::promise<response_type>
    resolve_async(components::component_type key)
    {
        return this->base_type::resolve_id_async
            (this->gid_, component_id_type(key));
    }
    
    response_type resolve(components::component_type key,
                          error_code& ec = throws)
    {
        return this->base_type::resolve_id
            (this->gid_, component_id_type(key), ec);
    }

    lcos::promise<response_type>
    resolve_async(component_name_type const& key)
    { return this->base_type::resolve_name_async(this->gid_, key); }
    
    response_type resolve(component_name_type const& key,
                          error_code& ec = throws)
    { return this->base_type::resolve_name(this->gid_, key, ec); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::promise<response_type>
    unbind_async(component_name_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    response_type unbind(component_name_type const& key,
                         error_code& ec = throws)
    { return this->base_type::unbind(this->gid_, key, ec); }
};            

}}

#endif // HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

