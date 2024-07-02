////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <atomic>
#include <cstdint>
#include <cstdlib>

template <typename T>
void test_atomic()
{
    std::atomic<T> a;
    if (!a.is_lock_free())
        std::exit(-1);
    a.store(T{});
    T i = a.load();
    (void)i;
}

struct uint128_type
{
    std::uint64_t left;
    std::uint64_t right;
};

int main()
{
    test_atomic<uint128_type>();
}
