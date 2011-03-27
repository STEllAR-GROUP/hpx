////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/util/lightweight_test.hpp>

namespace { struct aligned_16 { char a; } __attribute__ ((aligned(16))); }

int main()
{
    // Some GNU-esque compilers don't define this, but do 16-byte alignment
    // properly, so don't test this if it's not defined.
    #if defined(__BIGGEST_ALIGNMENT__)
      HPX_TEST(__BIGGEST_ALIGNMENT__ >= 16);
    #endif
    HPX_TEST_EQ(sizeof(aligned_16), 16);
    return hpx::util::report_errors();
}

