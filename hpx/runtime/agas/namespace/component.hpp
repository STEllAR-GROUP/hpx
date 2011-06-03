////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6)
#define HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/component.hpp>

namespace hpx { namespace agas 
{

// TODO: add error code parameters
template <typename Database>
struct component_namespace :
    components::client_base<
        component_namespace<Database>,
        stubs::component_namespace<Database>
    >
{
    // {{{ nested types 
    typedef components::client_base<
        component_namespace<Database>,
        stubs::component_namespace<Database>
    > base_type; 

    typedef server::component_namespace<Database> server_type;
    
    typedef typename server_type::component_name_type component_name_type;
    typedef typename server_type::component_id_type component_id_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    // }}}

    explicit component_namespace(naming::id_type const& id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::future_value<component_id_type>
    bind_async(component_name_type const& key, prefix_type prefix)
    { return this->base_type::bind_prefix_async(this->gid_, key, prefix); }

    component_id_type
    bind(component_name_type const& key, prefix_type prefix)
    { return this->base_type::bind_prefix(this->gid_, key, prefix); }
    
    lcos::future_value<component_id_type>
    bind_async(component_name_type const& key)
    { return this->base_type::bind_name_async(this->gid_, key); }
    
    component_id_type bind(component_name_type const& key)
    { return this->base_type::bind_name(this->gid_, key); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve_id and resolve_name interface - note the clever use of overloads
    // in the client.
    lcos::future_value<prefixes_type>
    resolve_async(component_id_type key)
    { return this->base_type::resolve_id_async(this->gid_, key); }
    
    prefixes_type resolve(component_id_type key)
    { return this->base_type::resolve_id(this->gid_, key); }

    lcos::future_value<component_id_type>
    resolve_async(component_name_type const& key)
    { return this->base_type::resolve_name_async(this->gid_, key); }
    
    component_id_type resolve(component_name_type const& key)
    { return this->base_type::resolve_name(this->gid_, key); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::future_value<bool>
    unbind_async(component_name_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    bool unbind(component_name_type const& key)
    { return this->base_type::unbind(this->gid_, key); }
};            

}}

#endif // HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

