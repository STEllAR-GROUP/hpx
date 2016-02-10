//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/detail/bootstrap_service_client.hpp>

namespace hpx { namespace agas { namespace detail
{
    void bootstrap_service_client::set_local_locality(naming::gid_type const& g)
    {
        data_.primary_ns_server_.set_local_locality(g);
    }

    response bootstrap_service_client::service(
        request const& req
        , threads::thread_priority priority
        , error_code& ec
        )
    {
        if (req.get_action_code() & primary_ns_service)
            return data_.primary_ns_server_.service(req, ec);

        if (req.get_action_code() & component_ns_service)
            return data_.component_ns_server_.service(req, ec);

        if (req.get_action_code() & symbol_ns_service)
            return data_.symbol_ns_server_.service(req, ec);

        if (req.get_action_code() & locality_ns_service)
            return data_.locality_ns_server_.service(req, ec);

        HPX_THROWS_IF(ec, bad_action_code
            , "addressing_service::service"
            , "invalid action code encountered in request")
            return response();
    }

    std::vector<response> bootstrap_service_client::bulk_service(
        std::vector<request> const& reqs
        , error_code& ec
        )
    {
        return data_.primary_ns_server_.bulk_service(reqs, ec);
    }

    void bootstrap_service_client::register_counter_types()
    {
        data_.register_counter_types();
    }

    void bootstrap_service_client::register_server_instance(
            boost::uint32_t locality_id)
    {
        std::string str("locality#" +
            boost::lexical_cast<std::string>(locality_id) + "/");
        return data_.register_server_instance(str.c_str());
    }

    bool bootstrap_service_client::unregister_server(
        request const& req
        , threads::thread_priority priority
        , error_code& ec)
    {
        data_.unregister_server_instance(ec);

        if (ec)
            return false;

        response rep = data_.locality_ns_server_.service(req, ec);

        if (ec || (success != rep.get_status()))
            return false;

        return true;
    }

    hpx::agas::response bootstrap_service_client::service_primary(
        request const& req
        , error_code& ec)
    {
        return data_.primary_ns_server_.service(req, ec);
    }

    hpx::agas::response bootstrap_service_client::service_component(
        request const& req
        , threads::thread_priority priority
        , error_code& ec)
    {
        return data_.component_ns_server_.service(req, ec);
    }

    hpx::agas::response bootstrap_service_client::service_locality(
        request const& req
        , threads::thread_priority priority
        , error_code& ec)
    {
        return data_.locality_ns_server_.service(req, ec);
    }
}}}

