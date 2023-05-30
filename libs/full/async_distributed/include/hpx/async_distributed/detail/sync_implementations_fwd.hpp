//  Copyright (c) 2018-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_local/sync_fwd.hpp>
#include <hpx/naming_base/id_type.hpp>

namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename... Ts>
    typename hpx::traits::extract_action_t<Action>::local_result_type sync_impl(
        Launch&& policy, hpx::id_type const& id, Ts&&... vs);
}    // namespace hpx::detail
