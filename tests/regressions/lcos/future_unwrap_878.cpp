//  Copyright 2013 (c) Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #878: `future::unwrap`
// triggers assertion

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <exception>
#include <utility>

int main()
{
    hpx::lcos::local::promise<hpx::future<int> > promise;
    hpx::future<hpx::future<int> > future = promise.get_future();
    try
    {
        //promise.set_value(42);
        throw hpx::bad_parameter;
    } catch(...) {
        promise.set_exception(std::current_exception());
    }
    HPX_TEST(future.has_exception());

    hpx::future<int> inner(std::move(future));
    HPX_TEST(inner.has_exception());

    return hpx::util::report_errors();
}
