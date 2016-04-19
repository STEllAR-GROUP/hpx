// Copyright (c) 2012 Vinay C Amatya
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3)
#define HPX_Sg7J0M91KTJ7FTMezA1gNjwaA2LCgq70PgnGNvX3

#include <hpx/config.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace performance_counters { namespace memory
{
    // returns virtual memory value
    boost::uint64_t read_psm_virtual(bool);

    // returns resident memory value
    boost::uint64_t read_psm_resident(bool);
}}}

#endif
