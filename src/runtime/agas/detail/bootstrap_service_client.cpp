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

    response bootstrap_service_client::service_primary_ns(request const& req, error_code& ec)
    {
        return data_.primary_ns_server_.service(req, ec);
    }

    /*response bootstrap_service_client::service_component_ns(request const& req, error_code& ec)
    {
        return data_.component_ns_server_.service(req, ec);
    }*/

    response bootstrap_service_client::service_symbol_ns(request const& req, error_code& ec)
    {
        return data_.symbol_ns_server_.service(req, ec);
    }

    /*response bootstrap_service_client::service_locality_ns(request const& req, error_code& ec)
    {
        return data_.locality_ns_server_.service(req, ec);
    }*/
}}}

