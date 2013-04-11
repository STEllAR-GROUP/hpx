//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/batch_environment.hpp>

int main()
{
    hpx::util::batch_environment env(true);

    std::cout << env.init_from_environment("");
}
