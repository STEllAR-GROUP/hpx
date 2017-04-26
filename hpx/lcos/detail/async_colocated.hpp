//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM)
#define HPX_LCOS_ASYNC_COLOCATED_FEB_01_2014_0105PM

#include <hpx/config.hpp>
#include <hpx/lcos/async_continue_fwd.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/detail/async_colocated_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/agas/primary_namespace.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>
#include <hpx/util/unique_function.hpp>

#include <type_traits>
#include <utility>

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
                    hpx::util::functional::extract_locality(
                        hpx::util::detail::placeholder<2ul> const&
                      , hpx::id_type&&
                    )
                >
              , Ts...
            >
            type;
    };
}}

#define HPX_REGISTER_ASYNC_COLOCATED_DECLARATION(Action, Name)                \
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(                            \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::async_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::async_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >,                                                                \
            hpx::util::unused_type                                            \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

#define HPX_REGISTER_ASYNC_COLOCATED(Action, Name)                            \
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION(                                        \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::async_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::async_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >,                                                                \
            hpx::util::unused_type                                            \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
/**/

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Action>::remote_result_type
        >::type>
    async_colocated(naming::id_type const& gid, Ts&&... vs)
    {
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return detail::async_continue_r<action_type, remote_result_type>(
            util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<Ts>(vs)...)
                ), service_target, gid.get_gid());
    }

    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::traits::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs)
    {
        return async_colocated<Derived>(gid, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename ...Ts>
    typename std::enable_if<
        traits::is_continuation<Continuation>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type
        >
    >::type
    async_colocated(Continuation && cont,
        naming::id_type const& gid, Ts&&... vs)
    {
        // Attach the requested action as a continuation to a resolve_async
        // call on the locality responsible for the target gid.
        naming::id_type service_target(
            agas::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);

        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
        remote_result_type;
        typedef agas::server::primary_namespace::colocate_action action_type;

        using util::placeholders::_2;
        return detail::async_continue_r<action_type, remote_result_type>(
            util::functional::async_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                      , std::forward<Ts>(vs)...)
                  , std::forward<Continuation>(cont))
              , service_target, gid.get_gid());
    }

    template <
        typename Continuation,
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    typename std::enable_if<
        traits::is_continuation<Continuation>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Derived>::remote_result_type
            >::type
        >
    >::type
    async_colocated(
        Continuation && cont
      , hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs)
    {
        return async_colocated<Derived>(std::forward<Continuation>(cont), gid,
            std::forward<Ts>(vs)...);
    }
}}

#endif
