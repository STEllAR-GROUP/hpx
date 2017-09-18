//  Copyright (c) 2017 Element-126
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/lcos/local/channel.hpp>
#include <hpx/lcos/channel.hpp>
#include <hpx/util/lightweight_test.hpp>

HPX_REGISTER_CHANNEL(int);

int main()
{
    {
        hpx::lcos::local::channel<int> ch;
        ch.set(0);
        ch.close(true);
    }

    {
        hpx::lcos::channel<int> ch (hpx::find_here());
        ch.set(0);
        ch.close(true);
    }

    return hpx::util::report_errors();
}
