////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <filesystem>

int main()
{
    std::filesystem::path p = std::filesystem::current_path();
    (void) p;
    return 0;
}
