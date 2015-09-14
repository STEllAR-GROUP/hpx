//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/batch_environment.hpp>

#include <iostream>

int main()
{
    std::vector<std::string> nodelist;
    hpx::util::batch_environment env(nodelist, true);

    for (std::string const& s: nodelist)
        std::cout << s << "\n";
}
