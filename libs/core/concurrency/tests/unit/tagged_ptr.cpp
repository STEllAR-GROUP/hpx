//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>

#include <limits>

void tagged_ptr_test()
{
    using namespace hpx::lockfree::detail;
    int a(1), b(2);

    using tag_t = tagged_ptr<int>::tag_t;
    constexpr tag_t max_tag = (std::numeric_limits<tag_t>::max)();

    {
        tagged_ptr<int> i(&a, 0);
        tagged_ptr<int> j(&b, 1);

        i = j;

        HPX_TEST_EQ(i.get_ptr(), &b);
        HPX_TEST_EQ(i.get_tag(), 1);
    }

    {
        tagged_ptr<int> i(&a, 0);
        tagged_ptr<int> j(i);

        HPX_TEST_EQ(i.get_ptr(), j.get_ptr());
        HPX_TEST_EQ(i.get_tag(), j.get_tag());
    }

    {
        tagged_ptr<int> i(&a, 0);
        HPX_TEST_EQ(i.get_tag() + 1, i.get_next_tag());
    }

    {
        tagged_ptr<int> j(&a, max_tag);
        HPX_TEST_EQ(j.get_next_tag(), 0);
    }

    {
        tagged_ptr<int> j(&a, max_tag - 1);
        HPX_TEST_EQ(j.get_next_tag(), max_tag);
    }
}

int main()
{
    tagged_ptr_test();

    return hpx::util::report_errors();
}
