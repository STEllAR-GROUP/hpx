//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/container_algorithms.hpp>
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/parallel/segmented_algorithm.hpp>
#endif

namespace hpx {
    using hpx::parallel::lexicographical_compare;
    using hpx::parallel::max_element;
    using hpx::parallel::min_element;
    using hpx::parallel::minmax_element;
    using hpx::parallel::partition;
    using hpx::parallel::partition_copy;
    using hpx::parallel::remove_copy;
    using hpx::parallel::remove_copy_if;
    using hpx::parallel::replace;
    using hpx::parallel::replace_copy;
    using hpx::parallel::replace_copy_if;
    using hpx::parallel::replace_if;
    using hpx::parallel::reverse;
    using hpx::parallel::reverse_copy;
    using hpx::parallel::rotate;
    using hpx::parallel::rotate_copy;
    using hpx::parallel::sort;
    using hpx::parallel::stable_partition;
    using hpx::parallel::stable_sort;
    using hpx::parallel::swap_ranges;
    using hpx::parallel::unique;
    using hpx::parallel::unique_copy;
}    // namespace hpx
