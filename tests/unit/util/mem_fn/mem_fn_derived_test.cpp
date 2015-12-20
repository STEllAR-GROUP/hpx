#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

// Taken from the Boost.Bind library
//
//  mem_fn_derived_test.cpp - tests mem_fn.hpp with derived objects
//
//  Copyright (c) 2001, 2002 Peter Dimov and Multi Media Ltd.
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/mem_fn.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

struct B
{
    mutable unsigned int hash;

    B(): hash(0) {}

    int f0() { f1(17); return 0; }
    int g0() const { g1(17); return 0; }

    int f1(int a1) { hash = (hash * 17041 + a1) % 32768; return 0; }
    int g1(int a1) const { hash = (hash * 17041 + a1 * 2) % 32768; return 0; }

    int f2(int a1, int a2) { f1(a1); f1(a2); return 0; }
    int g2(int a1, int a2) const { g1(a1); g1(a2); return 0; }

    int f3(int a1, int a2, int a3) { f2(a1, a2); f1(a3); return 0; }
    int g3(int a1, int a2, int a3) const { g2(a1, a2); g1(a3); return 0; }

    int f4(int a1, int a2, int a3, int a4) { f3(a1, a2, a3); f1(a4); return 0; }
    int g4(int a1, int a2, int a3, int a4) const { g3(a1, a2, a3); g1(a4); return 0; }

    int f5(int a1, int a2, int a3, int a4, int a5)
        { f4(a1, a2, a3, a4); f1(a5); return 0; }
    int g5(int a1, int a2, int a3, int a4, int a5)
        const { g4(a1, a2, a3, a4); g1(a5); return 0; }

    int f6(int a1, int a2, int a3, int a4, int a5, int a6)
        { f5(a1, a2, a3, a4, a5); f1(a6); return 0; }
    int g6(int a1, int a2, int a3, int a4, int a5, int a6)
        const { g5(a1, a2, a3, a4, a5); g1(a6); return 0; }

    int f7(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
        { f6(a1, a2, a3, a4, a5, a6); f1(a7); return 0; }
    int g7(int a1, int a2, int a3, int a4, int a5, int a6, int a7)
        const { g6(a1, a2, a3, a4, a5, a6); g1(a7); return 0; }

    int f8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
        { f7(a1, a2, a3, a4, a5, a6, a7); f1(a8); return 0; }
    int g8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8)
        const { g7(a1, a2, a3, a4, a5, a6, a7); g1(a8); return 0; }
};

struct X: public B
{
};

int main()
{
    X x;

    X const & rcx = x;
    X const * pcx = &x;

    boost::shared_ptr<X> sp(new X);

    hpx::util::mem_fn(&X::f0)(x);
    hpx::util::mem_fn(&X::f0)(&x);
    hpx::util::mem_fn(&X::f0)(sp);

    hpx::util::mem_fn(&X::g0)(x);
    hpx::util::mem_fn(&X::g0)(rcx);
    hpx::util::mem_fn(&X::g0)(&x);
    hpx::util::mem_fn(&X::g0)(pcx);
    hpx::util::mem_fn(&X::g0)(sp);

    hpx::util::mem_fn(&X::f1)(x, 1);
    hpx::util::mem_fn(&X::f1)(&x, 1);
    hpx::util::mem_fn(&X::f1)(sp, 1);

    hpx::util::mem_fn(&X::g1)(x, 1);
    hpx::util::mem_fn(&X::g1)(rcx, 1);
    hpx::util::mem_fn(&X::g1)(&x, 1);
    hpx::util::mem_fn(&X::g1)(pcx, 1);
    hpx::util::mem_fn(&X::g1)(sp, 1);

    hpx::util::mem_fn(&X::f2)(x, 1, 2);
    hpx::util::mem_fn(&X::f2)(&x, 1, 2);
    hpx::util::mem_fn(&X::f2)(sp, 1, 2);

    hpx::util::mem_fn(&X::g2)(x, 1, 2);
    hpx::util::mem_fn(&X::g2)(rcx, 1, 2);
    hpx::util::mem_fn(&X::g2)(&x, 1, 2);
    hpx::util::mem_fn(&X::g2)(pcx, 1, 2);
    hpx::util::mem_fn(&X::g2)(sp, 1, 2);

    hpx::util::mem_fn(&X::f3)(x, 1, 2, 3);
    hpx::util::mem_fn(&X::f3)(&x, 1, 2, 3);
    hpx::util::mem_fn(&X::f3)(sp, 1, 2, 3);

    hpx::util::mem_fn(&X::g3)(x, 1, 2, 3);
    hpx::util::mem_fn(&X::g3)(rcx, 1, 2, 3);
    hpx::util::mem_fn(&X::g3)(&x, 1, 2, 3);
    hpx::util::mem_fn(&X::g3)(pcx, 1, 2, 3);
    hpx::util::mem_fn(&X::g3)(sp, 1, 2, 3);

    hpx::util::mem_fn(&X::f4)(x, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::f4)(&x, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::f4)(sp, 1, 2, 3, 4);

    hpx::util::mem_fn(&X::g4)(x, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::g4)(rcx, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::g4)(&x, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::g4)(pcx, 1, 2, 3, 4);
    hpx::util::mem_fn(&X::g4)(sp, 1, 2, 3, 4);

    hpx::util::mem_fn(&X::f5)(x, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::f5)(&x, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::f5)(sp, 1, 2, 3, 4, 5);

    hpx::util::mem_fn(&X::g5)(x, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::g5)(rcx, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::g5)(&x, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::g5)(pcx, 1, 2, 3, 4, 5);
    hpx::util::mem_fn(&X::g5)(sp, 1, 2, 3, 4, 5);

    hpx::util::mem_fn(&X::f6)(x, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::f6)(&x, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::f6)(sp, 1, 2, 3, 4, 5, 6);

    hpx::util::mem_fn(&X::g6)(x, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::g6)(rcx, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::g6)(&x, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::g6)(pcx, 1, 2, 3, 4, 5, 6);
    hpx::util::mem_fn(&X::g6)(sp, 1, 2, 3, 4, 5, 6);

    hpx::util::mem_fn(&X::f7)(x, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::f7)(&x, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::f7)(sp, 1, 2, 3, 4, 5, 6, 7);

    hpx::util::mem_fn(&X::g7)(x, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::g7)(rcx, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::g7)(&x, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::g7)(pcx, 1, 2, 3, 4, 5, 6, 7);
    hpx::util::mem_fn(&X::g7)(sp, 1, 2, 3, 4, 5, 6, 7);

    hpx::util::mem_fn(&X::f8)(x, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::f8)(&x, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::f8)(sp, 1, 2, 3, 4, 5, 6, 7, 8);

    hpx::util::mem_fn(&X::g8)(x, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::g8)(rcx, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::g8)(&x, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::g8)(pcx, 1, 2, 3, 4, 5, 6, 7, 8);
    hpx::util::mem_fn(&X::g8)(sp, 1, 2, 3, 4, 5, 6, 7, 8);

    HPX_TEST(hpx::util::mem_fn(&X::hash)(x) == 17610);
    HPX_TEST(hpx::util::mem_fn(&X::hash)(sp) == 2155);

    return hpx::util::report_errors();
}
