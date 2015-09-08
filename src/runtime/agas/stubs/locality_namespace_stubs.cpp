////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/apply.hpp>
#include <hpx/runtime/agas/stubs/locality_namespace.hpp>

namespace hpx { namespace agas { namespace stubs
{
template <typename Result>
lcos::future<Result> locality_namespace::service_async(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    )
{
    typedef server_type::service_action action_type;

    lcos::packaged_action<action_type, Result> p;
    p.apply_p(launch::async, gid, priority, req);
    return p.get_future();
}

template lcos::future<response> locality_namespace::service_async<response>(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

template lcos::future<std::map<naming::gid_type, parcelset::endpoints_type> >
locality_namespace::service_async<std::map<naming::gid_type,
    parcelset::endpoints_type> >(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

template lcos::future<parcelset::endpoints_type>
locality_namespace::service_async<parcelset::endpoints_type>(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

template lcos::future<std::vector<boost::uint32_t> >
locality_namespace::service_async<std::vector<boost::uint32_t> >(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

template lcos::future<boost::uint32_t>
locality_namespace::service_async<boost::uint32_t>(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    );

void locality_namespace::service_non_blocking(
    naming::id_type const& gid
  , request const& req
  , threads::thread_priority priority
    )
{
    typedef server_type::service_action action_type;
    hpx::apply_p<action_type>(gid, priority, req);
}

lcos::future<std::vector<response> > locality_namespace::bulk_service_async(
    naming::id_type const& gid
  , std::vector<request> const& reqs
  , threads::thread_priority priority
    )
{
    typedef server_type::bulk_service_action action_type;

    lcos::packaged_action<action_type> p;
    p.apply_p(launch::async, gid, priority, reqs);
    return p.get_future();
}

void locality_namespace::bulk_service_non_blocking(
   naming::id_type const& gid
  , std::vector<request> const& reqs
  , threads::thread_priority priority
    )
{
    typedef server_type::bulk_service_action action_type;
    hpx::apply_p<action_type>(gid, priority, reqs);
}

}}}

