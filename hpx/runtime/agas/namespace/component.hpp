////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E425A0E2_3462_4C0F_ADB2_854881DBE6F1)
#define HPX_E425A0E2_3462_4C0F_ADB2_854881DBE6F1

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/stubs/component.hpp>

namespace hpx { namespace agas 
{

template <typename Database>
struct component_namespace
  : client_base<
      component_namespace<Database>,
      stubs::component_namespace<Database>
    >
{
    typedef client_base<
        component_namespace<Database>,
        stubs::basic_namespace<Tag>
    > base_type;

    typedef server::component_namespace<Database> server_type;

    typedef typename server_type::component_name_type component_name_type;
    typedef typename server_type::component_id_type component_id_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;

    explicit component_namespace(naming::id_type const& id = naming::invalid_id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<component_id_type>
    bind_async(component_name_type const& key, prefix_type prefix)
    { return this->base_type::bind_async(this->gid_, key, prefix); }

    component_id_type
    bind(component_name_type const& key, prefix_type prefix)
    { return this->base_type::bind(this->gid_, key, prefix); }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<prefixes_type>
    resolve_id_async(component_id_type key)
    { return this->base_type::resolve_id_async(this->gid_, key); }
    
    prefixes_type resolve_id(component_id_type key)
    { return this->base_type::resolve_id(this->gid_, key); }

    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<component_id_type>
    resolve_name_async(component_name_type const& key)
    { return this->base_type::resolve_name_async(this->gid_, key); }
    
    component_id_type resolve_name(component_name_type const& key)
    { return this->base_type::resolve_name(this->gid_, key); }
    
    ///////////////////////////////////////////////////////////////////////////
    lcos::future_value<void>
    unbind_async(component_name_type const& key)
    { return this->base_type::unbind_async(this->gid_, key); }
    
    void unbind(component_name_type const& key)
    { return this->base_type::unbind(this->gid_, key); }
};            

}}}

#endif // HPX_E425A0E2_3462_4C0F_ADB2_854881DBE6F1

