//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2013-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/iostreams/manipulators.hpp>

namespace hpx {
    // hpx::flush
    iostreams::flush_type flush = iostreams::flush_type();
    // hpx::endl
    iostreams::endl_type endl = iostreams::endl_type();
    // hpx::async_flush
    iostreams::async_flush_type async_flush = iostreams::async_flush_type();
    // hpx::async_endl
    iostreams::async_endl_type async_endl = iostreams::async_endl_type();
}    // namespace hpx
