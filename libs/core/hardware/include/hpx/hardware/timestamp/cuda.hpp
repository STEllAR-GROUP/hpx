////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2021 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/local/config.hpp>

#include <cstdint>

namespace hpx { namespace util { namespace hardware {

    HPX_DEVICE inline std::uint64_t timestamp_cuda()
    {
        std::uint64_t cur;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(cur));
        return cur;
    }

}}}    // namespace hpx::util::hardware
