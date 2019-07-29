// Copyright (c) 2012 Vinay C Amatya
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3)
#define HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx { namespace performance_counters { namespace memory
{
    // returns virtual memory value
    std::uint64_t read_psm_virtual(bool);

    // returns resident memory value
    std::uint64_t read_psm_resident(bool);

#if defined(__linux) || defined(linux) || defined(linux__) || defined(__linux__) \
 || defined(HPX_WINDOWS)
    // returns total available memory
    std::uint64_t read_total_mem_avail(bool);
#endif
}}}

#endif
