#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_dm3_test.cpp - data members (regression 1.31 - 1.32)
//
//  Copyright (c) 2005 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/bind.hpp>

namespace placeholders = hpx::util::placeholders;

#include <iostream>

#include <hpx/util/lightweight_test.hpp>
#include <utility>

int main()
{
    typedef std::pair<int, int> pair_type;

    pair_type pair( 10, 20 );

    int const & x = hpx::util::bind( &pair_type::first, placeholders::_1 )( pair );

    HPX_TEST( &pair.first == &x );

    return hpx::util::report_errors();
}
