//  Copyright 2013 (c) Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #878: `future::unwrap`
// triggers assertion

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <exception>
#include <utility>

int hpx_main()
{
    hpx::promise<hpx::future<int>> promise;
    hpx::future<hpx::future<int>> future = promise.get_future();
    std::exception_ptr p;
    try
    {
        //promise.set_value(42);
        throw hpx::error::bad_parameter;
    }
    catch (...)
    {
        p = std::current_exception();
    }
    HPX_TEST(p);
    promise.set_exception(std::move(p));
    HPX_TEST(future.has_exception());

    hpx::future<int> inner(std::move(future));
    HPX_TEST(inner.has_exception());

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
