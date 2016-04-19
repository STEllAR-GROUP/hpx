//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/unwrapped.hpp>

#include <vector>

void noop(){}

int main()
{
    std::vector<hpx::future<void>> fs;
    hpx::util::unwrapped(&noop)(fs);
}
