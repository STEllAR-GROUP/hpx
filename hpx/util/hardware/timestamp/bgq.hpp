////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_UTIL_HARDWARE_TIMESTAMP_BGQ_HPP)
#define HPX_UTIL_HARDWARE_TIMESTAMP_BGQ_HPP

#if defined(__bgq__)

// Hardware cycle-accurate timer on BGQ.
// see https://wiki.alcf.anl.gov/parts/index.php/Blue_Gene/Q#High-Resolution_Timers

#include <hwi/include/bqc/A2_inlines.h>

namespace hpx { namespace util { namespace hardware
{

inline boost::uint64_t timestamp()
{
    return GetTimeBase();
}

}}}

#endif

#endif // HPX_UTIL_HARDWARE_TIMESTAMP_LINUX_GENERIC_HPP

