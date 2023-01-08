//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/naming_base/id_type.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    template <typename Action, typename Cont, typename... Ts>
    bool post_continue(Cont&& cont, hpx::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename... Ts>
    bool post_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, hpx::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    bool post_continue(
        hpx::id_type const& cont, hpx::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    bool post_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& cont, hpx::id_type const& gid, Ts&&... vs);
}    // namespace hpx
