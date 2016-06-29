//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/stubs/symbol_namespace.hpp>
#include <hpx/runtime/agas/detail/hosted_service_client.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    void hosted_data_type::register_counter_types()
    {
        server::primary_namespace::register_counter_types();
        server::primary_namespace::register_global_counter_types();
        server::symbol_namespace::register_counter_types();
        server::symbol_namespace::register_global_counter_types();
    }

    void hosted_data_type::register_server_instance(char const* servicename
        , boost::uint32_t locality_id)
    {
        primary_ns_server_.register_server_instance(servicename, locality_id);
        symbol_ns_server_.register_server_instance(servicename, locality_id);
    }

    void hosted_data_type::unregister_server_instance(error_code& ec)
    {
        primary_ns_server_.unregister_server_instance(ec);
        if (!ec) symbol_ns_server_.unregister_server_instance(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void hosted_service_client::set_local_locality(
        naming::gid_type const& g)
    {
        data_.primary_ns_server_.set_local_locality(g);
    }

    response hosted_service_client::service(
        request const& req
      , threads::thread_priority priority
      , error_code& ec)
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
          , "invalid action code encountered in request");

        return response();
    }

    void hosted_service_client::register_counter_types()
    {
        data_.register_counter_types();
    }

    void hosted_service_client::register_server_instance(
            boost::uint32_t locality_id)
    {
        std::string str("locality#" + std::to_string(locality_id) + "/");
        data_.register_server_instance(str.c_str(), locality_id);
    }

    bool hosted_service_client::unregister_server(
        request const& req
      , threads::thread_priority priority
      , error_code& ec)
    {
        data_.unregister_server_instance(ec);

        if (ec)
            return false;

        response rep = data_.locality_ns_.service(req, priority, ec);

        if (ec || (success != rep.get_status()))
            return false;

        return true;
    }

    response hosted_service_client::service_primary(
        request const& req
      , error_code& ec)
    {
        return data_.primary_ns_server_.service(req, ec);
    }

    std::vector<response> hosted_service_client::service_primary_bulk(
        std::vector<request> const& reqs
      , error_code& ec)
    {
        return data_.primary_ns_server_.bulk_service(reqs, ec);
    }

    response hosted_service_client::service_component(
        request const& req
      , threads::thread_priority priority
      , error_code& ec)
    {
        return data_.component_ns_.service(req, priority, ec);
    }

    response hosted_service_client::service_locality(
        request const& req
      , threads::thread_priority priority
      , error_code& ec)
    {
        return data_.locality_ns_.service(req, priority, ec);
    }

    response hosted_service_client::service_symbol(
        request const& req
      , threads::thread_priority priority
      , std::string const& name
      , error_code& ec)
    {
        return stubs::symbol_namespace::service(name, req, priority, ec);
    }

    future<parcelset::endpoints_type> hosted_service_client::get_endpoints(
        request const& req
      , threads::thread_priority priority
      , error_code& ec)
    {
        return data_.locality_ns_.service_async<parcelset::endpoints_type>(
            req, priority);
    }
}}}

