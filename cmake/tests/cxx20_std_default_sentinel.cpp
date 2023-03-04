//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iterator>

int main()
{
    [[maybe_unused]] auto sent1 = std::default_sentinel;
    [[maybe_unused]] auto sent2 = std::default_sentinel_t{};
    return 0;
}
