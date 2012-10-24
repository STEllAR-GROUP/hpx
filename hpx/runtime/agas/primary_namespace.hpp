////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/include/client.hpp>
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
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return this->base_type::service(this->get_gid(), req, priority, ec);
    }

    void service_non_blocking(
        request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        this->base_type::service_non_blocking(this->get_gid(), req, priority);
    }

    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return this->base_type::bulk_service(this->get_gid(), reqs, priority, ec);
    }

    void bulk_service_non_blocking(
        std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        this->base_type::bulk_service_non_blocking(this->get_gid(), reqs, priority);
    }

    bool route(
        parcelset::parcel const& p
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return this->base_type::route(this->get_gid(), p, priority, ec);
    }

};

}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

