//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/naming_base.hpp>

#include <hpx/supervision_dispatch/server/registry.hpp>

#include <cstddef>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    ///////////////////////////////////////////////////////////////////////////
    class HPX_SUPERVISION_DISPATCH_EXPORT registry
      : public hpx::components::client_base<registry, server::registry>
    {
        using base_type =
            hpx::components::client_base<registry, server::registry>;

    public:
        explicit registry(hpx::id_type const& target_locality);
        /* implicit */ registry(hpx::future<hpx::id_type>&& f);
    };
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
