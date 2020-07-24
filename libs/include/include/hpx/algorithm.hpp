//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/container_algorithms.hpp>

namespace hpx {
    using hpx::parallel::adjacent_find;
    using hpx::parallel::all_of;
    using hpx::parallel::any_of;
    using hpx::parallel::count;
    using hpx::parallel::count_if;
    using hpx::parallel::equal;
    using hpx::parallel::fill;
    using hpx::parallel::fill_n;
    using hpx::parallel::find;
    using hpx::parallel::find_end;
    using hpx::parallel::find_first_of;
    using hpx::parallel::find_if;
    using hpx::parallel::find_if_not;
    using hpx::parallel::for_each;
    using hpx::parallel::for_each_n;
    using hpx::parallel::for_loop;
    using hpx::parallel::for_loop_n;
    using hpx::parallel::for_loop_n_strided;
    using hpx::parallel::for_loop_strided;
    using hpx::parallel::generate;
    using hpx::parallel::generate_n;
    using hpx::parallel::includes;
    using hpx::parallel::inplace_merge;
    using hpx::parallel::is_heap;
    using hpx::parallel::is_heap_until;
    using hpx::parallel::is_partitioned;
    using hpx::parallel::is_sorted;
    using hpx::parallel::is_sorted_until;
    using hpx::parallel::lexicographical_compare;
    using hpx::parallel::max_element;
    using hpx::parallel::merge;
    using hpx::parallel::min_element;
    using hpx::parallel::minmax_element;
    using hpx::parallel::mismatch;
    using hpx::parallel::move;
    using hpx::parallel::none_of;
    using hpx::parallel::partition;
    using hpx::parallel::partition_copy;
    using hpx::parallel::remove;
    using hpx::parallel::remove_copy;
    using hpx::parallel::remove_copy_if;
    using hpx::parallel::remove_if;
    using hpx::parallel::replace;
    using hpx::parallel::replace_copy;
    using hpx::parallel::replace_copy_if;
    using hpx::parallel::replace_if;
    using hpx::parallel::reverse;
    using hpx::parallel::reverse_copy;
    using hpx::parallel::rotate;
    using hpx::parallel::rotate_copy;
    using hpx::parallel::search;
    using hpx::parallel::search_n;
    using hpx::parallel::set_difference;
    using hpx::parallel::set_intersection;
    using hpx::parallel::set_symmetric_difference;
    using hpx::parallel::set_union;
    using hpx::parallel::sort;
    using hpx::parallel::stable_partition;
    using hpx::parallel::swap_ranges;
    using hpx::parallel::unique;
    using hpx::parallel::unique_copy;
}    // namespace hpx
