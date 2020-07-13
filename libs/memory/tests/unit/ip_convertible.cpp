//  wp_convertible_test.cpp
//
//  Copyright (c) 2008 Peter Dimov
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt

#include <hpx/config.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/testing.hpp>

//
struct W
{
};

void intrusive_ptr_add_ref(W*) {}

void intrusive_ptr_release(W*) {}

struct X : public virtual W
{
};

struct Y : public virtual W
{
};

struct Z : public X
{
};

int f(hpx::intrusive_ptr<X>)
{
    return 1;
}

int f(hpx::intrusive_ptr<Y>)
{
    return 2;
}

int main()
{
    HPX_TEST_EQ(1, f(hpx::intrusive_ptr<Z>()));
    return hpx::util::report_errors();
}
