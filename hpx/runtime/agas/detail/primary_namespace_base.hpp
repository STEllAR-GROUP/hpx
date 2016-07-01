//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_PRIMARY_NAMESPACE_BASE_JUN_30_2016_0119PM)
#define HPX_AGAS_PRIMARY_NAMESPACE_BASE_JUN_30_2016_0119PM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/basic_action.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/fixed_component_base.hpp>

#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct HPX_EXPORT primary_namespace_base
      : components::fixed_component_base<primary_namespace_base>
    {
        typedef components::fixed_component_base<primary_namespace_base> base_type;

        primary_namespace_base()
          : base_type(HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB)
        {}

        virtual ~primary_namespace_base() {}

        ///////////////////////////////////////////////////////////////////////
        response remote_service(request const& req)
        {
            return service(req, throws);
        }

        HPX_DEFINE_COMPONENT_ACTION(primary_namespace_base, remote_service,
            service_action);

        std::vector<response> remote_bulk_service(
            std::vector<request> const& reqs)
        {
            return bulk_service(reqs, throws);
        }

        HPX_DEFINE_COMPONENT_ACTION(primary_namespace_base, remote_bulk_service,
            bulk_service_action);

        response remote_route(parcelset::parcel && p)
        {
            return route(std::move(p));
        }

        HPX_DEFINE_COMPONENT_ACTION(primary_namespace_base, remote_route,
            route_action);

        ///////////////////////////////////////////////////////////////////////
        virtual response service(request const& req, error_code& ec) = 0;
        virtual std::vector<response> bulk_service(
            std::vector<request> const& reqs, error_code& ec) = 0;
        virtual response route(parcelset::parcel && p) = 0;
    };
}}}

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::detail::primary_namespace_base::service_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::detail::primary_namespace_base::bulk_service_action)

HPX_ACTION_USES_MEDIUM_STACK(
    hpx::agas::detail::primary_namespace_base::route_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::detail::primary_namespace_base::service_action,
    primary_namespace_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::detail::primary_namespace_base::bulk_service_action,
    primary_namespace_bulk_service_action)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::agas::detail::primary_namespace_base::route_action,
    primary_namespace_route_action)

#endif
