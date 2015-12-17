#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_not_test.cpp - operator!
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

template<class F, class A1, class R> void test( F f, A1 a1, R r )
{
    HPX_TEST( f(a1) == r );
}

bool f( bool v )
{
    return v;
}

int g( int v )
{
    return v;
}

int main()
{
    test( !hpx::util::bind( f, true ), 0, !f( true ) );
    test( !hpx::util::bind( g, placeholders::_1 ), 5, !g( 5 ) );
    test( hpx::util::bind( f, !hpx::util::bind( f, true ) ), 0, f( !f( true ) ) );
    test( hpx::util::bind( f, !hpx::util::bind( f, placeholders::_1 ) ),
        true, f( !f( true ) ) );

    return hpx::util::report_errors();
}
