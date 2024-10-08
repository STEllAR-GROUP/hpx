//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>

namespace hpx::util::detail {

    extern HPX_CORE_EXPORT void (*reinit_register)(
        std::function<void()> const& construct,
        std::function<void()> const& destruct);

    // Invoke all globally registered construction functions
    extern HPX_CORE_EXPORT void (*reinit_construct)();

    // Invoke all globally registered destruction functions
    extern HPX_CORE_EXPORT void (*reinit_destruct)();
}    // namespace hpx::util::detail
