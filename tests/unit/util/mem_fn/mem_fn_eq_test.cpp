#include <hpx/config.hpp>

#if defined(HPX_MSVC)
#pragma warning(disable: 4786)  // identifier truncated in debug info
#pragma warning(disable: 4710)  // function not inlined
#pragma warning(disable: 4711)  // function selected for automatic inline expansion
#pragma warning(disable: 4514)  // unreferenced inline removed
#endif

// Taken from the Boost.Bind library
//
//  mem_fn_eq_test.cpp - boost::mem_fn equality operator
//
//  Copyright (c) 2004 Peter Dimov
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/util/mem_fn.hpp>

#include <iostream>

#include <hpx/util/lightweight_test.hpp>

struct X
{
    int dm_1;
    int dm_2;

    // 0

    int mf0_1() { return 0; }
    int mf0_2() { return 0; }

    int cmf0_1() const { return 0; }
    int cmf0_2() const { return 0; }

    void mf0v_1() {}
    void mf0v_2() {}

    void cmf0v_1() const {}
    void cmf0v_2() const {}

    // 1

    int mf1_1(int) { return 0; }
    int mf1_2(int) { return 0; }

    int cmf1_1(int) const { return 0; }
    int cmf1_2(int) const { return 0; }

    void mf1v_1(int) {}
    void mf1v_2(int) {}

    void cmf1v_1(int) const {}
    void cmf1v_2(int) const {}

    // 2

    int mf2_1(int, int) { return 0; }
    int mf2_2(int, int) { return 0; }

    int cmf2_1(int, int) const { return 0; }
    int cmf2_2(int, int) const { return 0; }

    void mf2v_1(int, int) {}
    void mf2v_2(int, int) {}

    void cmf2v_1(int, int) const {}
    void cmf2v_2(int, int) const {}

    // 3

    int mf3_1(int, int, int) { return 0; }
    int mf3_2(int, int, int) { return 0; }

    int cmf3_1(int, int, int) const { return 0; }
    int cmf3_2(int, int, int) const { return 0; }

    void mf3v_1(int, int, int) {}
    void mf3v_2(int, int, int) {}

    void cmf3v_1(int, int, int) const {}
    void cmf3v_2(int, int, int) const {}

    // 4

    int mf4_1(int, int, int, int) { return 0; }
    int mf4_2(int, int, int, int) { return 0; }

    int cmf4_1(int, int, int, int) const { return 0; }
    int cmf4_2(int, int, int, int) const { return 0; }

    void mf4v_1(int, int, int, int) {}
    void mf4v_2(int, int, int, int) {}

    void cmf4v_1(int, int, int, int) const {}
    void cmf4v_2(int, int, int, int) const {}

    // 5

    int mf5_1(int, int, int, int, int) { return 0; }
    int mf5_2(int, int, int, int, int) { return 0; }

    int cmf5_1(int, int, int, int, int) const { return 0; }
    int cmf5_2(int, int, int, int, int) const { return 0; }

    void mf5v_1(int, int, int, int, int) {}
    void mf5v_2(int, int, int, int, int) {}

    void cmf5v_1(int, int, int, int, int) const {}
    void cmf5v_2(int, int, int, int, int) const {}

    // 6

    int mf6_1(int, int, int, int, int, int) { return 0; }
    int mf6_2(int, int, int, int, int, int) { return 0; }

    int cmf6_1(int, int, int, int, int, int) const { return 0; }
    int cmf6_2(int, int, int, int, int, int) const { return 0; }

    void mf6v_1(int, int, int, int, int, int) {}
    void mf6v_2(int, int, int, int, int, int) {}

    void cmf6v_1(int, int, int, int, int, int) const {}
    void cmf6v_2(int, int, int, int, int, int) const {}

    // 7

    int mf7_1(int, int, int, int, int, int, int) { return 0; }
    int mf7_2(int, int, int, int, int, int, int) { return 0; }

    int cmf7_1(int, int, int, int, int, int, int) const { return 0; }
    int cmf7_2(int, int, int, int, int, int, int) const { return 0; }

    void mf7v_1(int, int, int, int, int, int, int) {}
    void mf7v_2(int, int, int, int, int, int, int) {}

    void cmf7v_1(int, int, int, int, int, int, int) const {}
    void cmf7v_2(int, int, int, int, int, int, int) const {}

    // 8

    int mf8_1(int, int, int, int, int, int, int, int) { return 0; }
    int mf8_2(int, int, int, int, int, int, int, int) { return 0; }

    int cmf8_1(int, int, int, int, int, int, int, int) const { return 0; }
    int cmf8_2(int, int, int, int, int, int, int, int) const { return 0; }

    void mf8v_1(int, int, int, int, int, int, int, int) {}
    void mf8v_2(int, int, int, int, int, int, int, int) {}

    void cmf8v_1(int, int, int, int, int, int, int, int) const {}
    void cmf8v_2(int, int, int, int, int, int, int, int) const {}

};

int main()
{
    HPX_TEST( hpx::util::mem_fn(&X::dm_1) == hpx::util::mem_fn(&X::dm_1) );
    HPX_TEST( hpx::util::mem_fn(&X::dm_1) != hpx::util::mem_fn(&X::dm_2) );

    // 0

    HPX_TEST( hpx::util::mem_fn(&X::mf0_1) == hpx::util::mem_fn(&X::mf0_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf0_1) != hpx::util::mem_fn(&X::mf0_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf0_1) == hpx::util::mem_fn(&X::cmf0_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf0_1) != hpx::util::mem_fn(&X::cmf0_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf0v_1) == hpx::util::mem_fn(&X::mf0v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf0v_1) != hpx::util::mem_fn(&X::mf0v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf0v_1) == hpx::util::mem_fn(&X::cmf0v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf0v_1) != hpx::util::mem_fn(&X::cmf0v_2) );

    // 1

    HPX_TEST( hpx::util::mem_fn(&X::mf1_1) == hpx::util::mem_fn(&X::mf1_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf1_1) != hpx::util::mem_fn(&X::mf1_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf1_1) == hpx::util::mem_fn(&X::cmf1_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf1_1) != hpx::util::mem_fn(&X::cmf1_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf1v_1) == hpx::util::mem_fn(&X::mf1v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf1v_1) != hpx::util::mem_fn(&X::mf1v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf1v_1) == hpx::util::mem_fn(&X::cmf1v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf1v_1) != hpx::util::mem_fn(&X::cmf1v_2) );

    // 2

    HPX_TEST( hpx::util::mem_fn(&X::mf2_1) == hpx::util::mem_fn(&X::mf2_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf2_1) != hpx::util::mem_fn(&X::mf2_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf2_1) == hpx::util::mem_fn(&X::cmf2_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf2_1) != hpx::util::mem_fn(&X::cmf2_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf2v_1) == hpx::util::mem_fn(&X::mf2v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf2v_1) != hpx::util::mem_fn(&X::mf2v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf2v_1) == hpx::util::mem_fn(&X::cmf2v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf2v_1) != hpx::util::mem_fn(&X::cmf2v_2) );

    // 3

    HPX_TEST( hpx::util::mem_fn(&X::mf3_1) == hpx::util::mem_fn(&X::mf3_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf3_1) != hpx::util::mem_fn(&X::mf3_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf3_1) == hpx::util::mem_fn(&X::cmf3_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf3_1) != hpx::util::mem_fn(&X::cmf3_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf3v_1) == hpx::util::mem_fn(&X::mf3v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf3v_1) != hpx::util::mem_fn(&X::mf3v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf3v_1) == hpx::util::mem_fn(&X::cmf3v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf3v_1) != hpx::util::mem_fn(&X::cmf3v_2) );

    // 4

    HPX_TEST( hpx::util::mem_fn(&X::mf4_1) == hpx::util::mem_fn(&X::mf4_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf4_1) != hpx::util::mem_fn(&X::mf4_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf4_1) == hpx::util::mem_fn(&X::cmf4_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf4_1) != hpx::util::mem_fn(&X::cmf4_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf4v_1) == hpx::util::mem_fn(&X::mf4v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf4v_1) != hpx::util::mem_fn(&X::mf4v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf4v_1) == hpx::util::mem_fn(&X::cmf4v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf4v_1) != hpx::util::mem_fn(&X::cmf4v_2) );

    // 5

    HPX_TEST( hpx::util::mem_fn(&X::mf5_1) == hpx::util::mem_fn(&X::mf5_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf5_1) != hpx::util::mem_fn(&X::mf5_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf5_1) == hpx::util::mem_fn(&X::cmf5_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf5_1) != hpx::util::mem_fn(&X::cmf5_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf5v_1) == hpx::util::mem_fn(&X::mf5v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf5v_1) != hpx::util::mem_fn(&X::mf5v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf5v_1) == hpx::util::mem_fn(&X::cmf5v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf5v_1) != hpx::util::mem_fn(&X::cmf5v_2) );

    // 6

    HPX_TEST( hpx::util::mem_fn(&X::mf6_1) == hpx::util::mem_fn(&X::mf6_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf6_1) != hpx::util::mem_fn(&X::mf6_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf6_1) == hpx::util::mem_fn(&X::cmf6_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf6_1) != hpx::util::mem_fn(&X::cmf6_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf6v_1) == hpx::util::mem_fn(&X::mf6v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf6v_1) != hpx::util::mem_fn(&X::mf6v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf6v_1) == hpx::util::mem_fn(&X::cmf6v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf6v_1) != hpx::util::mem_fn(&X::cmf6v_2) );

    // 7

    HPX_TEST( hpx::util::mem_fn(&X::mf7_1) == hpx::util::mem_fn(&X::mf7_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf7_1) != hpx::util::mem_fn(&X::mf7_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf7_1) == hpx::util::mem_fn(&X::cmf7_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf7_1) != hpx::util::mem_fn(&X::cmf7_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf7v_1) == hpx::util::mem_fn(&X::mf7v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf7v_1) != hpx::util::mem_fn(&X::mf7v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf7v_1) == hpx::util::mem_fn(&X::cmf7v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf7v_1) != hpx::util::mem_fn(&X::cmf7v_2) );

    // 8

    HPX_TEST( hpx::util::mem_fn(&X::mf8_1) == hpx::util::mem_fn(&X::mf8_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf8_1) != hpx::util::mem_fn(&X::mf8_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf8_1) == hpx::util::mem_fn(&X::cmf8_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf8_1) != hpx::util::mem_fn(&X::cmf8_2) );

    HPX_TEST( hpx::util::mem_fn(&X::mf8v_1) == hpx::util::mem_fn(&X::mf8v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::mf8v_1) != hpx::util::mem_fn(&X::mf8v_2) );

    HPX_TEST( hpx::util::mem_fn(&X::cmf8v_1) == hpx::util::mem_fn(&X::cmf8v_1) );
    HPX_TEST( hpx::util::mem_fn(&X::cmf8v_1) != hpx::util::mem_fn(&X::cmf8v_2) );


    return hpx::util::report_errors();
}
