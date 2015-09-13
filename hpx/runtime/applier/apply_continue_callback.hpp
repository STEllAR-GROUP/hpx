//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_RUNTIME_APPLIER_APPLY_CONTINUE_CALLBACK_MAR_09_2014_1207PM)
#define HPX_RUNTIME_APPLIER_APPLY_CONTINUE_CALLBACK_MAR_09_2014_1207PM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/apply.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename Callback,
        typename ...Ts>
    bool apply_continue_cb(Cont&& cont, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;

        return apply_cb<Action>(
            hpx::actions::typed_continuation<result_type>(std::forward<Cont>(cont)),
            gid, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename Callback, typename ...Ts>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_continue_cb<Derived>(std::forward<Cont>(cont), gid,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    bool apply_continue_cb(naming::id_type const& cont,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename action_type::result_type result_type;

        return apply_cb<Action>(
            hpx::actions::typed_continuation<result_type>(cont, make_continuation()),
            gid, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    bool apply_continue_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& cont, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        return apply_continue_cb<Derived>(cont, gid,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}

#endif
