////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D)
#define HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>

namespace hpx { namespace agas { namespace stubs
{

struct component_namespace
{
    typedef server::component_namespace server_type;

    typedef server_type::component_id_type component_id_type;
    typedef server_type::prefixes_type prefixes_type;

    template <
        typename Result
    >
    static lcos::promise<Result, response> service_async(
        naming::id_type const& gid
      , request const& req
        )
    {
        typedef server_type::service_action action_type;
        return lcos::eager_future<action_type, Result>(gid, req);
    }

    static response service(
        naming::id_type const& gid
      , request const& req
      , error_code& ec = throws
        )
    {
        return service_async<response>(gid, req).get(ec);
    }
};

}}}

#endif // HPX_85B78E29_DD30_4603_8EF5_29EFB32FD10D

