//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/detail/hosted_service_client.hpp>

namespace hpx { namespace agas { namespace detail
{
    void hosted_service_client::set_local_locality(naming::gid_type const& g)
    {
        data_.primary_ns_server_.set_local_locality(g);
    }

    response hosted_service_client::service(
        request const& req
      , threads::thread_priority priority
      , error_code& ec
        )
    {
        if (req.get_action_code() & primary_ns_service)
            return data_.primary_ns_server_.service(req, ec);

        if (req.get_action_code() & component_ns_service)
            return data_.component_ns_.service(req, priority, ec);

        if (req.get_action_code() & symbol_ns_service)
            return data_.symbol_ns_server_.service(req, ec);

        if (req.get_action_code() & locality_ns_service)
            return data_.locality_ns_.service(req, priority, ec);

        HPX_THROWS_IF(ec, bad_action_code
            , "addressing_service::service"
            , "invalid action code encountered in request")
            return response();
    }

    std::vector<response> hosted_service_client::bulk_service(
        std::vector<request> const& reqs
        , error_code& ec
        )
    {
        return data_.primary_ns_server_.bulk_service(reqs, ec);
    }
}}}

