//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STACKTRACES)
#include <hpx/debugging/backtrace/backtrace.hpp>
#else

#include <cstddef>
#include <string>

namespace hpx::util {

    class backtrace
    {
    };

    inline std::string trace(
        std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH)
    {
        return "";
    }
}    // namespace hpx::util

#endif
