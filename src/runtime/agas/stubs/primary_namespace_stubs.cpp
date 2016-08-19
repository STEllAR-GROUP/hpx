////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/apply.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/packaged_action.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace hpx { namespace agas { namespace stubs
{

template <typename Result>
lcos::future<Result> primary_namespace::service_async(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    )
{
    typedef server_type::service_action action_type;

    lcos::packaged_action<action_type, hpx::agas::response> p;
    lcos::future<agas::response> f = p.get_future();
    p.apply_p(gid, priority, req);
    return hpx::make_future<Result>(
            std::move(f),
            [](hpx::agas::response const& resp)
            {
                return agas::get_response_result<Result>::call(resp);
            }
        );
}

template lcos::future<response>
    primary_namespace::service_async<response>(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority
        );

template lcos::future<bool>
    primary_namespace::service_async<bool>(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority
        );

template lcos::future<std::int64_t>
    primary_namespace::service_async<std::int64_t>(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority
        );

template lcos::future<naming::id_type>
    primary_namespace::service_async<naming::id_type>(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority
        );

template lcos::future<naming::address>
    primary_namespace::service_async<naming::address>(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority
        );

template lcos::future<std::pair<naming::id_type, naming::address> >
    primary_namespace::service_async<std::pair<naming::id_type, naming::address> >(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

void primary_namespace::service_non_blocking(
    naming::id_type const& gid
  , request const& req
  , util::function_nonser<void(boost::system::error_code const&,
        parcelset::parcel const&)> const& f
  , threads::thread_priority priority
    )
{
    typedef server_type::service_action action_type;
    hpx::apply_p_cb<action_type>(gid, priority, f, req);
}

void primary_namespace::service_non_blocking(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    )
{
    typedef server_type::service_action action_type;
    hpx::apply_p<action_type>(gid, priority, req);
}

lcos::future<std::vector<response> >
    primary_namespace::bulk_service_async(
        naming::id_type const& gid
      , std::vector<request> reqs
      , threads::thread_priority priority
        )
{
    typedef server_type::bulk_service_action action_type;

    lcos::packaged_action<action_type> p;
    lcos::future<std::vector<response> > f = p.get_future();
    p.apply_p(gid, priority, std::move(reqs));
    return f;
}

void primary_namespace::bulk_service_non_blocking(
   naming::id_type const& gid
  , std::vector<request> reqs
  , threads::thread_priority priority
    )
{
    typedef server_type::bulk_service_action action_type;
    hpx::apply_p<action_type>(gid, priority, std::move(reqs));
}

}}}

