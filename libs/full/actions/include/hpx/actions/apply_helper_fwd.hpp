//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_select_direct_execution.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/threading_base/thread_init_data.hpp>

namespace hpx { namespace applier { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    void call_async(threads::thread_init_data&& data,
        naming::id_type const& target, naming::address::address_type lva,
        naming::address::component_type comptype,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Continuation, typename... Ts>
    void call_async(threads::thread_init_data&& data, Continuation&& cont,
        naming::id_type const& target, naming::address::address_type lva,
        naming::address::component_type comptype,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename... Ts>
    HPX_FORCEINLINE void call_sync(naming::address::address_type lva,
        naming::address::component_type comptype, Ts&&... vs);

    template <typename Action, typename Continuation, typename... Ts>
    HPX_FORCEINLINE void call_sync(Continuation&& cont,
        naming::address::address_type lva,
        naming::address::component_type comptype, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action,
        bool DirectExecute = Action::direct_execution::value>
    struct apply_helper;
}}}    // namespace hpx::applier::detail
