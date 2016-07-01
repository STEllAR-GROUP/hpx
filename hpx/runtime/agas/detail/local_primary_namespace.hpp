//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_CLIENT_LOCAL_PRIMARY_NAMESPACE_JUN_29_2106_0114PM)
#define HPX_AGAS_CLIENT_LOCAL_PRIMARY_NAMESPACE_JUN_29_2106_0114PM

#include <hpx/config.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/agas/detail/primary_namespace_counter_data.hpp>
#include <hpx/runtime/agas/detail/primary_namespace_base.hpp>

#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct local_primary_namespace : detail::primary_namespace_base
    {
        response service(request const& req, error_code& ec);

        std::vector<response> bulk_service(std::vector<request> const& reqs,
            error_code& ec);

        response route(parcelset::parcel && p);

    protected:
        response allocate(request const& req, error_code& ec);
        response bind_gid(request const& req, error_code& ec);
        response resolve_gid(request const& req, error_code& ec);
        response unbind_gid(request const& req, error_code& ec);

    private:
        detail::primary_namespace_counter_data counter_data_;
    };
}}}

#endif

