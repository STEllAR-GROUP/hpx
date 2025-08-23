//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \page hpx::distributed::promise
/// \headerfile hpx/future.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/futures/future_fwd.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/futures/traits/promise_remote_result.hpp>

#include <vector>

namespace hpx {

    /// \namespace lcos
    namespace lcos {
        class HPX_EXPORT base_lco;

        template <typename Result,
            typename RemoteResult =
                typename traits::promise_remote_result<Result>::type,
            typename ComponentType = traits::detail::managed_component_tag>
        class base_lco_with_value;

        template <typename ComponentType>
        class base_lco_with_value<void, void, ComponentType>;

        template <typename Action,
            typename Result = typename traits::promise_local_result<
                typename Action::remote_result_type>::type,
            bool DirectExecute = Action::direct_execution::value>
        class packaged_action;
    }    // namespace lcos

    namespace distributed {

        template <typename Result,
            typename RemoteResult =
                typename traits::promise_remote_result<Result>::type>
        class promise;
    }
}    // namespace hpx
