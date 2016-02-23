//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM

#include <hpx/config.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // This template meta function can be used to extract the action type, no
    // matter whether it got specified directly or by passing the
    // corresponding make_action<> specialization.
    template <typename Action, typename Enable = void>
    struct extract_action
    {
        typedef typename Action::derived_type type;
        typedef typename type::result_type result_type;
        typedef typename type::remote_result_type remote_result_type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // dispatch point used for async implementations
        template <typename Func, typename Enable = void>
        struct async_dispatch;

        // dispatch point used for async<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct async_action_dispatch;

        // dispatch point used for launch_policy implementations
        template <typename Action, typename Enable = void>
        struct async_launch_policy_dispatch;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto async(F && f, Ts &&... ts)
    ->  decltype(detail::async_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...
        ));

    template <typename F, typename ...Ts>
    HPX_FORCEINLINE auto async(F&& f, Ts&&... ts)
    ->  decltype(detail::async_dispatch<typename util::decay<F>::type>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...
        ));
}

#endif
