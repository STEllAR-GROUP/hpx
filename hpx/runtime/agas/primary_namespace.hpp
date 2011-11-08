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
    typedef components::client_base<primary_namespace, stubs::primary_namespace>
        base_type;

    typedef server::primary_namespace server_type;

    primary_namespace()
      : base_type(bootstrap_primary_namespace_id())
    {}

    explicit primary_namespace(naming::id_type const& id)
      : base_type(id)
    {}

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

