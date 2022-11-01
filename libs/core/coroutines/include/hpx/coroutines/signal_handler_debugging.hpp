//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>

namespace hpx::threads::coroutines {

    HPX_CORE_EXPORT extern bool attach_debugger_on_sigv;
    HPX_CORE_EXPORT extern bool diagnostics_on_terminate;
    HPX_CORE_EXPORT extern int exception_verbosity;
    HPX_CORE_EXPORT extern std::size_t trace_depth;
#if !defined(HPX_WINDOWS)
    HPX_CORE_EXPORT extern bool register_signal_handler;
#endif
}    // namespace hpx::threads::coroutines
