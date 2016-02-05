//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Parsa Amini
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_CLIENT_BASE_FEB_05_2016_1144AM)
#define HPX_AGAS_CLIENT_BASE_FEB_05_2016_1144AM

#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_data_type;
    struct hosted_data_type;

    struct agas_service_client
    {
        virtual ~agas_service_client() {}

        virtual void set_local_locality(naming::gid_type const& g) = 0;
        virtual response service_primary_ns(request const& req, error_code& ec) = 0;
        /*virtual response service_component_ns(request const& req, error_code& ec) = 0;*/
        virtual response service_symbol_ns(request const& req, error_code& ec) = 0;
        /*virtual response service_locality_ns(request const& req, error_code& ec) = 0;*/
    };
}}}

#endif

