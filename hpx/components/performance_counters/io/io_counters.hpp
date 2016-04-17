//  Copyright (c) 2015 Maciej Brodowicz
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PERFORMANCE_COUNTERS_IO_COUNTERS_IO_COUNTERS_201508111554)
#define PERFORMANCE_COUNTERS_IO_COUNTERS_IO_COUNTERS_201508111554

#include <hpx/config.hpp>

namespace hpx { namespace performance_counters { namespace io
{
    // returns number of bytes passed as an argument to read I/O operations
    boost::uint64_t get_pio_riss(bool);
    // returns number of bytes passed as an argument to write I/O operations
    boost::uint64_t get_pio_wiss(bool);
    // returns number of system calls resulting in read I/O operations
    boost::uint64_t get_pio_rsysc(bool);
    // returns number of system calls resulting in write I/O operations
    boost::uint64_t get_pio_wsysc(bool);
    // returns number of bytes transferred from storage
    boost::uint64_t get_pio_rstor(bool);
    // returns number of bytes transferred to storage
    boost::uint64_t get_pio_wstor(bool);
    // returns number of bytes transferred to storage that were later removed
    // due to truncation or deletion
    boost::uint64_t get_pio_wcanc(bool);
}}}

#endif // !defined(PERFORMANCE_COUNTERS_IO_COUNTERS_IO_COUNTERS_201508111554)
