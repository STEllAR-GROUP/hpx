//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/memory.hpp>

#include <hpx/parallel/container_memory.hpp>

namespace hpx {
    using hpx::parallel::uninitialized_copy;
    using hpx::parallel::uninitialized_copy_n;
    using hpx::parallel::uninitialized_default_construct;
    using hpx::parallel::uninitialized_default_construct_n;
    using hpx::parallel::uninitialized_fill;
    using hpx::parallel::uninitialized_fill_n;
    using hpx::parallel::uninitialized_move;
    using hpx::parallel::uninitialized_move_n;
    using hpx::parallel::uninitialized_value_construct;
    using hpx::parallel::uninitialized_value_construct_n;
}    // namespace hpx
