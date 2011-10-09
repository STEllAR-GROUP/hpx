////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6)
#define HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/component_namespace.hpp>

namespace hpx { namespace agas 
{

struct component_namespace :
    components::client_base<component_namespace, stubs::component_namespace>
{
    // {{{ nested types 
    typedef components::client_base<
        component_namespace, stubs::component_namespace
    > base_type; 

    typedef server::component_namespace server_type;
    
    typedef server_type::component_id_type component_id_type;
    typedef server_type::prefixes_type prefixes_type;
    // }}}

    component_namespace()
      : base_type(bootstrap_component_namespace_id())
    {}

    explicit component_namespace(naming::id_type const& id)
      : base_type(id)
    {}

    lcos::promise<response> service_async(
        request const& req 
        )
    {
        return this->base_type::service_async(this->gid_, req);
    }

    response service(
        request const& req 
      , error_code& ec = throws
        )
    {
        return this->base_type::service(this->gid_, req, ec);
    }
};            

}}

#endif // HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

