//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/actions.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/components_base.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/tracing.hpp>

#include <hpx/supervision_dispatch/export_definitions.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::supervision::server {

    ///////////////////////////////////////////////////////////////////////////
    class HPX_SUPERVISION_DISPATCH_EXPORT registry
      : public hpx::components::component_base<registry>
    {
    public:
        registry();
    };
}    // namespace hpx::supervision::server

#include <hpx/config/warnings_suffix.hpp>
