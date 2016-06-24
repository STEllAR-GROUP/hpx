////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_874FFB61_BEF5_4D46_B024_6DAAF81BACF1)
#define HPX_874FFB61_BEF5_4D46_B024_6DAAF81BACF1

#include <hpx/config.hpp>
#if defined(HPX_WINDOWS)

#include <boost/cstdint.hpp>
#include <intrin.h>
#include <windows.h>

namespace hpx { namespace util { namespace hardware
{
    inline boost::uint64_t timestamp()
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return static_cast<boost::uint64_t>(now.QuadPart);
    }
}}}

#endif

#endif // HPX_874FFB61_BEF5_4D46_B024_6DAAF81BACF1

