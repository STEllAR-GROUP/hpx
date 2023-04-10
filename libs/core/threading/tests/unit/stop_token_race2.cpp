//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <iostream>

///////////////////////////////////////////////////////////////////////////////
void test_callback_throw()
{
    // create stop_source
    hpx::stop_source ssrc;
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());

    // create stop_token from stop_source
    hpx::stop_token stok{ssrc.get_token()};
    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());

    // register callback
    bool cb1_called{false};
    bool cb2_called{false};

    auto cb = [&] {
        cb1_called = true;
        // throw
        throw "callback called";
    };

    hpx::stop_callback<decltype(cb)> cb1(stok, cb);

    HPX_TEST(ssrc.stop_possible());
    HPX_TEST(!ssrc.stop_requested());
    HPX_TEST(stok.stop_possible());
    HPX_TEST(!stok.stop_requested());
    HPX_TEST(!cb1_called);
    HPX_TEST(!cb2_called);

    // catch terminate() call:
    std::set_terminate([] {
        std::cout << "std::terminate called\n";
        std::exit(hpx::util::report_errors());
    });

    // request stop
    ssrc.request_stop();
    HPX_TEST(false);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // this test terminates execution
    test_callback_throw();

    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
