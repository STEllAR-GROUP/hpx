//  Copyright (c) 2007-2014 Hartmut Kaiser
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
    bool apply_continue(Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Cont, typename... Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    bool apply_continue(
        naming::id_type const& cont, naming::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    bool apply_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& cont, naming::id_type const& gid, Ts&&... vs);
}    // namespace hpx
