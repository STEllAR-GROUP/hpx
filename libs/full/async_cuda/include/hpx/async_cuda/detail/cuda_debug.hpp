//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2020 John Biddiscombe
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/debugging/print.hpp>

namespace hpx { namespace cuda { namespace experimental { namespace detail {
    using print_on = debug::enable_print<false>;
    static constexpr print_on cud_debug("CUDA");
}}}}    // namespace hpx::cuda::experimental::detail
