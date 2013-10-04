#include <hpx/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

// Taken from the Boost.Bind library
//
//  mem_fn_ref_test.cpp - reference_wrapper
//
//  Copyright (c) 2009 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt
//

#include <hpx/util/mem_fn.hpp>
#include <boost/ref.hpp>
#include <hpx/util/lightweight_test.hpp>

struct X
{
    int f()
    {
        return 1;
    }

    int g() const
    {
        return 2;
    }
};

int main()
{
    X x;

    HPX_TEST( hpx::util::mem_fn( &X::f )( boost::ref( x ) ) == 1 );
    HPX_TEST( hpx::util::mem_fn( &X::g )( boost::cref( x ) ) == 2 );

    return hpx::util::report_errors();
}
