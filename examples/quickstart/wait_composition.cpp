////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/tuple.hpp>

#include <iostream>

///////////////////////////////////////////////////////////////////////////////
struct cout_continuation
{
    typedef hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>>
        data_type;

    void operator()(hpx::future<data_type> data) const
    {
        data_type v = data.get();
        std::cout << hpx::get<0>(v).get() << "\n";
        std::cout << hpx::get<1>(v).get() << "\n";
        std::cout << hpx::get<2>(v).get() << "\n";
    }
};
///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::future<int> a = hpx::make_ready_future<int>(17);
        hpx::future<int> b = hpx::make_ready_future<int>(42);
        hpx::future<int> c = hpx::make_ready_future<int>(-1);

        hpx::when_all(a, b, c).then(cout_continuation());
    }

    return hpx::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::local::init(hpx_main, argc, argv);    // Initialize and run HPX.
}
