////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file BOOST_LICENSE_1_0.rst or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E293906B_B264_4CFF_93C4_B64B354E7BD6)
#define HPX_E293906B_B264_4CFF_93C4_B64B354E7BD6

#include <limits.h>

#include <boost/integer.hpp>

namespace hpx {

// If you get an error in this header, your platform doesn't have an integer
// type large enough to store a pointer; hpx needs such a type to function
// properly.
typedef boost::uint_t<sizeof(void*) * CHAR_BIT>::exact uintptr_t;

} // hpx

#endif // HPX_E293906B_B264_4CFF_93C4_B64B354E7BD6

