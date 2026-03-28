//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>

#include <hpx/execution_base/stdexec_forward.hpp>

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#elif defined(HPX_CLANG_VERSION)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include <exec/env.hpp>

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#elif defined(HPX_CLANG_VERSION)
#pragma clang diagnostic pop
#endif

namespace hpx::execution::experimental {
    HPX_CXX_CORE_EXPORT using exec::with_t;

    HPX_CXX_CORE_EXPORT using exec::with;
    HPX_CXX_CORE_EXPORT using exec::without;

    HPX_CXX_CORE_EXPORT using exec::make_env;
    HPX_CXX_CORE_EXPORT using exec::make_env_t;

    HPX_CXX_CORE_EXPORT using exec::write;
    HPX_CXX_CORE_EXPORT using exec::write_env;

    HPX_CXX_CORE_EXPORT using exec::read_with_default;
}    // namespace hpx::execution::experimental
