//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks that unwrapping does not attempt to make copies of
// noncopyable lvalues.

#include <hpx/pack_traversal/unwrap.hpp>

struct noncopyable
{
    noncopyable() = default;
    noncopyable(noncopyable&&) = default;
    noncopyable& operator=(noncopyable&&) = default;
    noncopyable(noncopyable const&) = delete;
    noncopyable& operator=(noncopyable const&) = delete;
};

int main()
{
    auto f = hpx::util::unwrapping([](noncopyable const&) {});
    noncopyable n{};
    f(n);
}
