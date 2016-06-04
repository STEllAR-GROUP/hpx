//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM)
#define HPX_LCOS_ASYNC_CALLBACK_FWD_MAR_30_2015_1122AM

#include <hpx/config.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/id_type.hpp>

#ifndef HPX_MSVC
#include <boost/utility/enable_if.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // dispatch point used for async_cb implementations
        template <typename Func, typename Enable = void>
        struct async_cb_dispatch;

        // dispatch point used for async_cb<Action> implementations
        template <typename Action, typename Func, typename Enable = void>
        struct async_cb_action_dispatch;
    }

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
    template <typename Action, typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto async_cb(F && f, Ts &&... ts)
    ->  decltype(detail::async_cb_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...));
}

#endif

