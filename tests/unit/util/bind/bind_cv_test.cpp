#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

//  Taken from the Boost.Bind library
//
//  bind_cv_test.cpp
//
//  Copyright (c) 2004 Peter Dimov
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

struct X
{
    // SGI-related compilers have odd compiler-synthesized ctors dtors
    #ifdef __PATHSCALE__
    X() {}
    ~X() {}
    #endif

    int operator()()
    {
        return 17041;
    }

    int operator()() const
    {
        return -17041;
    }

    int operator()(int x1)
    {
        return x1;
    }

    int operator()(int x1) const
    {
        return -x1;
    }

    int operator()(int x1, int x2)
    {
        return x1+x2;
    }

    int operator()(int x1, int x2) const
    {
        return -(x1+x2);
    }

    int operator()(int x1, int x2, int x3)
    {
        return x1+x2+x3;
    }

    int operator()(int x1, int x2, int x3) const
    {
        return -(x1+x2+x3);
    }

    int operator()(int x1, int x2, int x3, int x4)
    {
        return x1+x2+x3+x4;
    }

    int operator()(int x1, int x2, int x3, int x4) const
    {
        return -(x1+x2+x3+x4);
    }

    int operator()(int x1, int x2, int x3, int x4, int x5)
    {
        return x1+x2+x3+x4+x5;
    }

    int operator()(int x1, int x2, int x3, int x4, int x5) const
    {
        return -(x1+x2+x3+x4+x5);
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6)
    {
        return x1+x2+x3+x4+x5+x6;
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6) const
    {
        return -(x1+x2+x3+x4+x5+x6);
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7)
    {
        return x1+x2+x3+x4+x5+x6+x7;
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7) const
    {
        return -(x1+x2+x3+x4+x5+x6+x7);
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7, int x8)
    {
        return x1+x2+x3+x4+x5+x6+x7+x8;
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7, int x8) const
    {
        return -(x1+x2+x3+x4+x5+x6+x7+x8);
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7, int x8, int x9)
    {
        return x1+x2+x3+x4+x5+x6+x7+x8+x9;
    }

    int operator()(int x1, int x2, int x3, int x4, int x5,
        int x6, int x7, int x8, int x9) const
    {
        return -(x1+x2+x3+x4+x5+x6+x7+x8+x9);
    }
};

template<class F> void test(F f, int r)
{
    F const & cf = f;
    HPX_TEST( cf() == -r );
    HPX_TEST( f() == r );
}

int main()
{
    test( hpx::util::bind( X() ), 17041 );
    test( hpx::util::bind( X(), 1 ), 1 );
    test( hpx::util::bind( X(), 1, 2 ), 1+2 );
    test( hpx::util::bind( X(), 1, 2, 3 ), 1+2+3 );
    test( hpx::util::bind( X(), 1, 2, 3, 4 ), 1+2+3+4 );
    test( hpx::util::bind( X(), 1, 2, 3, 4, 5 ), 1+2+3+4+5 );
    test( hpx::util::bind( X(), 1, 2, 3, 4, 5, 6 ), 1+2+3+4+5+6 );
    test( hpx::util::bind( X(), 1, 2, 3, 4, 5, 6, 7 ), 1+2+3+4+5+6+7 );
    test( hpx::util::bind( X(), 1, 2, 3, 4, 5, 6, 7, 8 ), 1+2+3+4+5+6+7+8 );
    test( hpx::util::bind( X(), 1, 2, 3, 4, 5, 6, 7, 8, 9 ), 1+2+3+4+5+6+7+8+9 );

    return hpx::util::report_errors();
}
