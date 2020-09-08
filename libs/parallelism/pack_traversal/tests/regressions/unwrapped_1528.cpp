//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/futures/future.hpp>
#include <hpx/hpx.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <vector>

void noop() {}

int main()
{
    std::vector<hpx::future<void>> fs;
    hpx::util::unwrapping (&noop)(fs);
}
