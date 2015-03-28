//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM)
#define HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/agas/request.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/async_continue_fwd.hpp>
#include <hpx/lcos/async_colocated_fwd.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>

namespace hpx { namespace detail
{
    template <typename Tuple>
    struct async_colocated_bound_tuple;

    template <typename ...Ts>
    struct async_colocated_bound_tuple<util::tuple<Ts...> >
    {
        typedef
            util::tuple<
                hpx::util::detail::bound<
                    hpx::util::functional::extract_locality
                  , hpx::util::tuple<
                        hpx::util::detail::placeholder<2ul>
                      , hpx::id_type
                    >
                >
              , Ts...
            >
            type;
    };
}}

#define HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(Action, Name)                \
    HPX_UTIL_REGISTER_FUNCTION_DECLARATION(                                   \
        void (hpx::naming::id_type, hpx::agas::response)                      \
      , (hpx::util::functional::detail::async_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::async_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

#define HPX_REGISTER_ASYNC_COLOCATED(Action, Name)                            \
    HPX_UTIL_REGISTER_FUNCTION(                                               \
        void (hpx::naming::id_type, hpx::agas::response)                      \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::async_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_colocated(naming::id_type const& gid, Ts&&... vs)
    {
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef
            typename hpx::actions::extract_action<Action>::remote_result_type
        remote_result_type;
        typedef agas::server::primary_namespace::service_action action_type;

        using util::placeholders::_2;
        return detail::async_continue_r<action_type, remote_result_type>(
            util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<Ts>(vs)...)
                ), service_target, req);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs)
    {
        return async_colocated<Derived>(gid, std::forward<Ts>(vs)...);
    }
}

#endif
