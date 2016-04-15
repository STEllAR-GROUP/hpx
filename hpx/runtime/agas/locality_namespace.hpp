////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AGAS_LOCALITY_NAMESPACE_APR_03_2013_1139AM)
#define HPX_AGAS_LOCALITY_NAMESPACE_APR_03_2013_1139AM

#include <hpx/include/client.hpp>
#include <hpx/runtime/agas/stubs/locality_namespace.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <vector>

namespace hpx { namespace agas
{

struct locality_namespace
  : components::client_base<locality_namespace, stubs::locality_namespace>
{
    typedef components::client_base<locality_namespace, stubs::locality_namespace>
        base_type;

    typedef server::locality_namespace server_type;

    locality_namespace()
      : base_type(bootstrap_locality_namespace_id())
    {}

    explicit locality_namespace(naming::id_type const& id)
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

    template <typename Result>
    future<Result> service_async(
        request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        )
    {
        return this->base_type::service_async<Result>(this->get_id(), req, priority);
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

#endif

