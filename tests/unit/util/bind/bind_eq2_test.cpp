#include <hpx/hpx_init.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_eq2_test.cpp - hpx::util::bind equality operator
//
//  Copyright (c) 2004, 2005, 2009 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/bind.hpp>
#include <boost/function_equal.hpp>
#include <hpx/util/lightweight_test.hpp>

namespace placeholders = hpx::util::placeholders;

void f( int )
{
}

int g( int i )
{
    return i + 5;
}

template< class F > void test_self_equal( F f )
{
#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
    using hpx::function_equal;
#endif

    HPX_TEST( function_equal( f, f ) );
}

int main()
{
    test_self_equal( hpx::util::bind( f, placeholders::_1 ) );
    test_self_equal( hpx::util::bind( g, placeholders::_1 ) );
    test_self_equal( hpx::util::bind( f, hpx::util::bind( g, placeholders::_1 ) ) );

    return hpx::util::report_errors();
}
