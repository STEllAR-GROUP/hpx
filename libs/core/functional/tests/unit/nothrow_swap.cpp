//  Taken from the Boost.Function library

//  Copyright Douglas Gregor 2008.
//  Copyright 2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
// Boost.Function library

// For more information, see http://www.boost.org

#include <hpx/functional/function.hpp>
#include <hpx/modules/testing.hpp>

struct tried_to_copy
{
};

struct MaybeThrowOnCopy
{
    MaybeThrowOnCopy(int value = 0)
      : value(value)
    {
    }

    MaybeThrowOnCopy(const MaybeThrowOnCopy& other)
      : value(other.value)
    {
        if (throwOnCopy)
            throw tried_to_copy();
    }

    // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
    MaybeThrowOnCopy& operator=(const MaybeThrowOnCopy& other)
    {
        if (throwOnCopy)
            throw tried_to_copy();
        value = other.value;
        return *this;
    }

    int operator()()
    {
        return value;
    }

    int value;

    // Make sure that this function object doesn't trigger the
    // small-object optimization in Function.
    float padding[100];

    static bool throwOnCopy;
};

bool MaybeThrowOnCopy::throwOnCopy = false;

int main(int, char*[])
{
    hpx::util::function_nonser<int()> f;
    hpx::util::function_nonser<int()> g;

    MaybeThrowOnCopy::throwOnCopy = false;
    f = MaybeThrowOnCopy(1);
    g = MaybeThrowOnCopy(2);
    HPX_TEST_EQ(f(), 1);
    HPX_TEST_EQ(g(), 2);

    MaybeThrowOnCopy::throwOnCopy = true;
    f.swap(g);
    HPX_TEST_EQ(f(), 2);
    HPX_TEST_EQ(g(), 1);

    return hpx::util::report_errors();
}
