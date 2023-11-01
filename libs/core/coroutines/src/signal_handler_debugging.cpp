//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/coroutines/signal_handler_debugging.hpp>

#include <cstddef>

namespace hpx::threads::coroutines {

    bool attach_debugger_on_sigv = false;
    bool diagnostics_on_terminate = true;
    int exception_verbosity = 2;
#if defined(HPX_HAVE_STACKTRACES) && defined(HPX_HAVE_THREAD_BACKTRACE_DEPTH)
    std::size_t trace_depth = HPX_HAVE_THREAD_BACKTRACE_DEPTH;
#else
    std::size_t trace_depth = 0;
#endif
#if !defined(HPX_WINDOWS)
    bool register_signal_handler = true;
#endif
}    // namespace hpx::threads::coroutines
