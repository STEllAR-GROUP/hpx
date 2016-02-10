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

        virtual response service(
            request const& req
            , threads::thread_priority priority =
                threads::thread_priority_default
            , error_code& ec = throws
            ) = 0;

        virtual std::vector<response> bulk_service(
            std::vector<request> const& reqs
            , error_code& ec
            ) = 0;

        virtual void register_counter_types() = 0;

        virtual void register_server_instance(boost::uint32_t locality_id) = 0;

        virtual bool unregister_server(
            request const& req
            , threads::thread_priority priority =
                threads::thread_priority_default
            , error_code& ec = throws) = 0;

        virtual response service_primary(
            request const& req
            , error_code& ec = throws
            ) = 0;

        virtual response service_component(
            request const& req
            , threads::thread_priority priority =
            threads::thread_priority_default
            , error_code& ec = throws
            ) = 0;

        virtual response service_locality(
            request const& req
            , threads::thread_priority priority =
            threads::thread_priority_default
            , error_code& ec = throws
            ) = 0;
    };
}}}

#endif

