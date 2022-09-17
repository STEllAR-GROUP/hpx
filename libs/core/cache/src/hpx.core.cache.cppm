//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

export module hpx.core.cache;

export import <hpx/cache/entries/entry.hpp>;
export import <hpx/cache/entries/fifo_entry.hpp>;
export import <hpx/cache/entries/lfu_entry.hpp>;
export import <hpx/cache/entries/lru_entry.hpp>;
export import <hpx/cache/entries/size_entry.hpp>;
export import <hpx/cache/local_cache.hpp>;
export import <hpx/cache/lru_cache.hpp>;
export import <hpx/cache/policies/always.hpp>;
export import <hpx/cache/statistics/local_full_statistics.hpp>;
export import <hpx/cache/statistics/local_statistics.hpp>;
export import <hpx/cache/statistics/no_statistics.hpp>;
