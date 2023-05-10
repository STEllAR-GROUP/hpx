//  Copyright (c) 2014 Erik Schnetter
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <iostream>
#include <utility>

using namespace hpx;

future<void> nested_future()
{
    return make_ready_future();
}

int hpx_main()
{
    std::cout << "Starting...\n";

    future<future<void>> f1 = async(launch::deferred, &nested_future);

    future<void> f2(std::move(f1));
    f2.wait();

    std::cout << "Done.\n";
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
