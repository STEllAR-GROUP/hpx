////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iterator>

int main()
{
    constexpr bool b = std::disable_sized_sentinel_for<void, void>;
    (void) b;

    return 0;
}
