////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D)
#define HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

#include <hpx/config.hpp>
//#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/runtime/agas/response.hpp>

#include <vector>

namespace hpx { namespace agas {
    struct request;
    struct response;
namespace server
{
    struct component_namespace;
}
namespace stubs
{
    struct HPX_EXPORT component_namespace
    {
        typedef server::component_namespace server_type;
        typedef server::component_namespace server_component_type;

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

#endif // HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

