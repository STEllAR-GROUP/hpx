//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_RUNTIME_APPLIER_APPLY_CONTINUE_JAN_27_2013_0726PM)
#define HPX_RUNTIME_APPLIER_APPLY_CONTINUE_JAN_27_2013_0726PM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/action_support.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/traits/extract_action.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename ...Ts>
    bool apply_continue(Cont&& cont, naming::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::remote_result_type remote_result_type;
        typedef typename action_type::local_result_type local_result_type;

        return apply<Action>(
            hpx::actions::typed_continuation<
                local_result_type, remote_result_type>(std::forward<Cont>(cont)),
            gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename ...Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs)
    {
        return apply_continue<Derived>(std::forward<Cont>(cont), gid,
            std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    bool apply_continue(naming::id_type const& cont,
        naming::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::type action_type;
        typedef typename action_type::remote_result_type remote_result_type;
        typedef typename action_type::local_result_type local_result_type;

        return apply<Action>(
            hpx::actions::typed_continuation<
                local_result_type, remote_result_type>(cont, make_continuation()),
            gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& cont, naming::id_type const& gid, Ts&&... vs)
    {
        return apply_continue<Derived>(cont, gid, std::forward<Ts>(vs)...);
    }
}

#endif
