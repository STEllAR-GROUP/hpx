//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_distributed/applier/apply_continue_fwd.hpp>
#include <hpx/async_distributed/make_continuation.hpp>

#include <utility>

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename... Ts>
    bool apply_continue(Cont&& cont, id_type const& gid, Ts&&... vs)
    {
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using remote_result_type = typename action_type::remote_result_type;
        using local_result_type = typename action_type::local_result_type;

        return apply<Action>(hpx::actions::typed_continuation<local_result_type,
                                 remote_result_type>(std::forward<Cont>(cont)),
            gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename... Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, id_type const& gid, Ts&&... vs)
    {
        return apply_continue<Derived>(
            std::forward<Cont>(cont), gid, std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    bool apply_continue(id_type const& cont, id_type const& gid, Ts&&... vs)
    {
        using action_type = typename hpx::traits::extract_action<Action>::type;
        using remote_result_type = typename action_type::remote_result_type;
        using local_result_type = typename action_type::local_result_type;

        return apply<Action>(hpx::actions::typed_continuation<local_result_type,
                                 remote_result_type>(cont, make_continuation()),
            gid, std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        id_type const& cont, id_type const& gid, Ts&&... vs)
    {
        return apply_continue<Derived>(cont, gid, std::forward<Ts>(vs)...);
    }
}    // namespace hpx
