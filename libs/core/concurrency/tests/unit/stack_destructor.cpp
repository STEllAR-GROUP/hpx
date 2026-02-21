//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>

int g_instance_counter = 0;

struct tester
{
    tester()
    {
        ++g_instance_counter;
    }

    tester(tester const&)
    {
        ++g_instance_counter;
    }

    ~tester()
    {
        --g_instance_counter;
    }
};

void stack_instance_deleter_test()
{
    {
        hpx::lockfree::stack<tester> q(128);
        q.push(tester());
        q.push(tester());
        q.push(tester());
        q.push(tester());
        q.push(tester());
    }

    HPX_TEST(g_instance_counter == 0);
}

struct no_default_init_tester
{
    int value;

    explicit no_default_init_tester(int value)
      : value(value)
    {
        ++g_instance_counter;
    }

    no_default_init_tester(no_default_init_tester const& t)
    {
        value = t.value;

        ++g_instance_counter;
    }

    ~no_default_init_tester()
    {
        --g_instance_counter;
    }
};

void stack_instance_deleter_no_default_init_test()
{
    {
        hpx::lockfree::stack<no_default_init_tester> q(128);
        q.push(no_default_init_tester(1));
        q.push(no_default_init_tester(2));
        q.push(no_default_init_tester(3));
        q.push(no_default_init_tester(4));
        q.push(no_default_init_tester(5));
    }

    HPX_TEST(g_instance_counter == 0);
}

int main()
{
    stack_instance_deleter_test();
    stack_instance_deleter_no_default_init_test();

    return hpx::util::report_errors();
}
