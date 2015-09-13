//  Taken from the Boost.Bind library
//  protect_test.cpp
//
//  Copyright (c) 2009 Steven Watanabe
//  Copyright (c) 2013 Agustin Berge
//
// Distributed under the Boost Software License, Version 1.0.
//
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/protect.hpp>

namespace placeholders = hpx::util::placeholders;

#include <hpx/util/lightweight_test.hpp>

int f(int x)
{
    return x;
}

int& g(int& x)
{
    return x;
}

template<class T>
const T& constify(const T& arg)
{
    return arg;
}

int main()
{
    int i[9] = {0,1,2,3,4,5,6,7,8};

    // non-const

    // test nullary
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, 1))(), 1);

    // test lvalues

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0]), &i[0]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1]), &i[1]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2]), &i[2]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3]), &i[3]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3], i[4]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3], i[4]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3], i[4]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3], i[4]), &i[3]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_5))(i[0], i[1], i[2], i[3], i[4]), &i[4]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[3]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_5))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[4]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_6))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[5]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[3]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_5))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[4]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_6))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[5]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_7))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[6]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[3]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_5))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[4]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_6))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[5]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_7))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[6]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_8))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]), &i[7]);

    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[0]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[1]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[2]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_4))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[3]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_5))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[4]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_6))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[5]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_7))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[6]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_8))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[7]);
    HPX_TEST_EQ(&hpx::util::protect(hpx::util::bind(g,
        placeholders::_9))(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]), &i[8]);

    // test rvalues

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, placeholders::_1))(0), 0);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1), 1);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2), 2);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3), 3);
//
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3, 4), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3, 4), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3, 4), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3, 4), 3);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5))(0, 1, 2, 3, 4), 4);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3, 4, 5), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3, 4, 5), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3, 4, 5), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3, 4, 5), 3);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5))(0, 1, 2, 3, 4, 5), 4);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6))(0, 1, 2, 3, 4, 5), 5);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3, 4, 5, 6), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3, 4, 5, 6), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3, 4, 5, 6), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3, 4, 5, 6), 3);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5))(0, 1, 2, 3, 4, 5, 6), 4);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6))(0, 1, 2, 3, 4, 5, 6), 5);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7))(0, 1, 2, 3, 4, 5, 6), 6);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3, 4, 5, 6, 7), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3, 4, 5, 6, 7), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3, 4, 5, 6, 7), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3, 4, 5, 6, 7), 3);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5))(0, 1, 2, 3, 4, 5, 6, 7), 4);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6))(0, 1, 2, 3, 4, 5, 6, 7), 5);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7))(0, 1, 2, 3, 4, 5, 6, 7), 6);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_8))(0, 1, 2, 3, 4, 5, 6, 7), 7);

    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1))(0, 1, 2, 3, 4, 5, 6, 7, 8), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2))(0, 1, 2, 3, 4, 5, 6, 7, 8), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3))(0, 1, 2, 3, 4, 5, 6, 7, 8), 2);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4))(0, 1, 2, 3, 4, 5, 6, 7, 8), 3);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5))(0, 1, 2, 3, 4, 5, 6, 7, 8), 4);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6))(0, 1, 2, 3, 4, 5, 6, 7, 8), 5);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7))(0, 1, 2, 3, 4, 5, 6, 7, 8), 6);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_8))(0, 1, 2, 3, 4, 5, 6, 7, 8), 7);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f,
        placeholders::_9))(0, 1, 2, 3, 4, 5, 6, 7, 8), 8);

    // test mixed perfect forwarding
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, placeholders::_1))(i[0], 1), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, placeholders::_2))(i[0], 1), 1);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, placeholders::_1))(0, i[1]), 0);
    HPX_TEST_EQ(hpx::util::protect(hpx::util::bind(f, placeholders::_2))(0, i[1]), 1);

    // const

    // test nullary
    HPX_TEST_EQ(constify(constify(hpx::util::protect(hpx::util::bind(f, 1))))(), 1);

    // test lvalues
    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))))(i[0]), &i[0]);

    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))))(i[0], i[1]), &i[0]);
    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))))(i[0], i[1]), &i[1]);

    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1))))(i[0], i[1], i[2]), &i[0]);
    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2))))(i[0], i[1], i[2]), &i[1]);
    HPX_TEST_EQ(&constify(constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3))))(i[0], i[1], i[2]), &i[2]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3]), &i[3]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3], i[4]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3], i[4]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3], i[4]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3], i[4]), &i[3]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_5)))(i[0], i[1], i[2], i[3], i[4]), &i[4]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[3]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_5)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[4]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_6)))(i[0], i[1], i[2], i[3], i[4], i[5]), &i[5]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[3]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_5)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[4]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_6)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[5]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_7)))(i[0], i[1], i[2], i[3], i[4], i[5], i[6]), &i[6]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[3]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_5)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[4]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_6)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[5]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_7)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[6]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_8)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7]), &i[7]);

    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_1)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[0]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_2)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[1]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_3)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[2]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_4)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[3]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_5)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[4]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_6)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[5]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_7)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[6]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_8)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[7]);
    HPX_TEST_EQ(&constify(hpx::util::protect(hpx::util::bind(g,
        placeholders::_9)))(i[0], i[1], i[2], i[3], i[4], i[5],
            i[6], i[7], i[8]), &i[8]);

    // test rvalues

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0), 0);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1), 1);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2), 2);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3), 3);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3, 4), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3, 4), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3, 4), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3, 4), 3);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5)))(0, 1, 2, 3, 4), 4);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3, 4, 5), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3, 4, 5), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3, 4, 5), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3, 4, 5), 3);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5)))(0, 1, 2, 3, 4, 5), 4);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6)))(0, 1, 2, 3, 4, 5), 5);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3, 4, 5, 6), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3, 4, 5, 6), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3, 4, 5, 6), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3, 4, 5, 6), 3);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5)))(0, 1, 2, 3, 4, 5, 6), 4);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6)))(0, 1, 2, 3, 4, 5, 6), 5);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7)))(0, 1, 2, 3, 4, 5, 6), 6);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3, 4, 5, 6, 7), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3, 4, 5, 6, 7), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3, 4, 5, 6, 7), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3, 4, 5, 6, 7), 3);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5)))(0, 1, 2, 3, 4, 5, 6, 7), 4);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6)))(0, 1, 2, 3, 4, 5, 6, 7), 5);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7)))(0, 1, 2, 3, 4, 5, 6, 7), 6);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_8)))(0, 1, 2, 3, 4, 5, 6, 7), 7);

    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_3)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 2);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_4)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 3);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_5)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 4);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_6)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 5);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_7)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 6);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_8)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 7);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_9)))(0, 1, 2, 3, 4, 5, 6, 7, 8), 8);

    // test mixed perfect forwarding
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(i[0], 1), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(i[0], 1), 1);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_1)))(0, i[1]), 0);
    HPX_TEST_EQ(constify(hpx::util::protect(hpx::util::bind(f,
        placeholders::_2)))(0, i[1]), 1);

    return hpx::util::report_errors();
}
