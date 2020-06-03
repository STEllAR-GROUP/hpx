//  Copyright (c) 2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/batch_environments.hpp>

#include <iostream>
#include <string>
#include <vector>

int main()
{
    std::vector<std::string> nodelist;
    hpx::util::batch_environment env(nodelist, false, true);

    for (std::string const& s : nodelist)
        std::cout << s << "\n";

    return 0;
}
