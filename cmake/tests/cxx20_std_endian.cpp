////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <bit>

int main()
{
    if (std::endian::native == std::endian::big)
    {
    }
    else if (std::endian::native == std::endian::little)
    {
    }

    return 0;
}
