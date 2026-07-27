//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/runtime_components.hpp>
#include <hpx/modules/runtime_distributed.hpp>

#include <hpx/supervision_dispatch/registry.hpp>
#include <hpx/supervision_dispatch/server/registry.hpp>

#include <cstddef>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    registry::registry(hpx::id_type const& target_locality)
      : base_type(hpx::new_<server::registry>(target_locality))
    {
    }

    registry::registry(hpx::future<hpx::id_type>&& f)
      : base_type(HPX_MOVE(f))
    {
    }
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
