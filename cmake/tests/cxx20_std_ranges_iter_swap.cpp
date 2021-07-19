//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


// test for availability of std::ranges::iter_swap (C++ 20)

#include <iterator>
#include <vector>

int main()
{
    std::vector<int> buff1(1, 0);
    std::vector<int> buff2(1, 1);
    std::ranges::iter_swap(buff1.begin(), buff2.begin());

    return 0;
}
