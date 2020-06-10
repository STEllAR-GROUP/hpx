//  Copyright (c) 2019 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/lcos_local/composable_guard.hpp>

namespace hpx { namespace lcos_local {

    struct force_linking_helper
    {
        void (*free)(lcos::local::detail::guard_task* task);
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::lcos_local
