//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/naming_base/id_type.hpp>

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename... Ts>
    bool apply_colocated_cb(hpx::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename... Ts>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    bool apply_colocated_cb(Continuation&& cont, hpx::id_type const& gid,
        Callback&& cb, Ts&&... vs);

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename Callback, typename... Ts>
    bool apply_colocated_cb(Continuation&& cont,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& gid, Callback&& cb, Ts&&... vs);
}}    // namespace hpx::detail

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx {
    using hpx::detail::apply_colocated_cb;
}
#endif
