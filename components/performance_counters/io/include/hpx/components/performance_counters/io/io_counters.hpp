//  Copyright (c) 2015 Maciej Brodowicz
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx { namespace performance_counters { namespace io
{
    // returns number of bytes passed as an argument to read I/O operations
    std::uint64_t get_pio_riss(bool);
    // returns number of bytes passed as an argument to write I/O operations
    std::uint64_t get_pio_wiss(bool);
    // returns number of system calls resulting in read I/O operations
    std::uint64_t get_pio_rsysc(bool);
    // returns number of system calls resulting in write I/O operations
    std::uint64_t get_pio_wsysc(bool);
    // returns number of bytes transferred from storage
    std::uint64_t get_pio_rstor(bool);
    // returns number of bytes transferred to storage
    std::uint64_t get_pio_wstor(bool);
    // returns number of bytes transferred to storage that were later removed
    // due to truncation or deletion
    std::uint64_t get_pio_wcanc(bool);
}}}

