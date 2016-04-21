////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AGAS_LOCALITY_NAMESPACE_APR_03_2013_1138AM)
#define HPX_AGAS_LOCALITY_NAMESPACE_APR_03_2013_1138AM

#include <hpx/config.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>

#include <vector>

namespace hpx { namespace agas { namespace stubs
{

struct HPX_EXPORT locality_namespace
{
    typedef server::locality_namespace server_type;
    typedef server::locality_namespace server_component_type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    static lcos::future<Result> service_async(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void service_non_blocking(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static response service(
        naming::id_type const& gid
      , request const& req
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return service_async<response>(gid, req, priority).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future<std::vector<response> > bulk_service_async(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    /// Fire-and-forget semantics.
    ///
    /// \note This is placed out of line to avoid including applier headers.
    static void bulk_service_non_blocking(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
        );

    static std::vector<response> bulk_service(
        naming::id_type const& gid
      , std::vector<request> const& reqs
      , threads::thread_priority priority = threads::thread_priority_default
      , error_code& ec = throws
        )
    {
        return bulk_service_async(gid, reqs, priority).get(ec);
    }
};

}}}

#endif

