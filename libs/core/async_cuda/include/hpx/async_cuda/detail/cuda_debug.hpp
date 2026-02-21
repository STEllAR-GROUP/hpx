//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/debugging.hpp>

namespace hpx::cuda::experimental::detail {

    HPX_CXX_CORE_EXPORT using print_on = debug::enable_print<false>;
    HPX_CXX_CORE_EXPORT inline constexpr print_on cud_debug("CUDA");
}    // namespace hpx::cuda::experimental::detail
