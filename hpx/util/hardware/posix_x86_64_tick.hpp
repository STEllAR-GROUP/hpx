////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E0A61B00_A571_48EF_8516_ECB48CDDBC00)
#define HPX_E0A61B00_A571_48EF_8516_ECB48CDDBC00

#include <boost/cstdint.hpp>

extern "C" boost::uint64_t HPX_posix_x86_64_hardware_tick();

namespace hpx { namespace util { namespace hardware
{

inline boost::uint64_t tick() {
  return HPX_posix_x86_64_hardware_tick(); 
}

}}}

#endif // HPX_E0A61B00_A571_48EF_8516_ECB48CDDBC00

