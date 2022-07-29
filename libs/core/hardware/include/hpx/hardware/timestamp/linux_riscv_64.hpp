////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2022 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#if defined(__riscv)

#include <cstdint>

namespace hpx { namespace util { namespace hardware {

    // clang-format off
    HPX_HOST_DEVICE inline std::uint64_t timestamp()
    {
        std::uint64_t val = 0;
        __asm__ __volatile__(
                "rdtime %0;\n"
                : "=r"(val)
                :: );
        return val;
    }
    // clang-format on

}}}    // namespace hpx::util::hardware

#endif
