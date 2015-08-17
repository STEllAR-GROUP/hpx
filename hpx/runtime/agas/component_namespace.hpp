////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6)
#define HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

#include <hpx/include/client.hpp>
#include <hpx/runtime/agas/stubs/component_namespace.hpp>

namespace hpx { namespace agas
{

HPX_EXPORT naming::gid_type bootstrap_component_namespace_gid();
HPX_EXPORT naming::id_type bootstrap_component_namespace_id();

struct component_namespace
  : components::client_base<component_namespace, stubs::component_namespace>
{
    // {{{ nested types
    typedef components::client_base<
        component_namespace, stubs::component_namespace
    > base_type;

    typedef server::component_namespace server_type;

    component_namespace()
      : base_type(bootstrap_component_namespace_id())
    {}

    explicit component_namespace(naming::id_type const& id)
      : base_type(id)
    {}

    response service(
        request const& req
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return this->base_type::service(this->get_id(), req, priority, ec);
    }

    void service_non_blocking(
        request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        this->base_type::service_non_blocking(this->get_id(), req, priority);
    }

    std::vector<response> bulk_service(
        std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return this->base_type::bulk_service(this->get_id(), reqs, priority, ec);
    }

    void bulk_service_non_blocking(
        std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        this->base_type::bulk_service_non_blocking(this->get_id(), reqs, priority);
    }
};

}}

#endif // HPX_5ABE62AC_CDBC_4EAE_B01B_693CB5F2C0E6

