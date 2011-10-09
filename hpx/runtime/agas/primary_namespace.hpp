////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>

namespace hpx { namespace agas 
{

struct primary_namespace :
    components::client_base<primary_namespace, stubs::primary_namespace>
{
    // {{{ nested types 
    typedef components::client_base<primary_namespace, stubs::primary_namespace>
        base_type; 

    typedef server::primary_namespace server_type;

    typedef server_type::endpoint_type endpoint_type;
    typedef server_type::gva_type gva_type;
    typedef server_type::count_type count_type;
    typedef server_type::offset_type offset_type;
    typedef server_type::prefix_type prefix_type;
    // }}}

    primary_namespace()
      : base_type(bootstrap_primary_namespace_id())
    {}

    explicit primary_namespace(naming::id_type const& id)
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

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

