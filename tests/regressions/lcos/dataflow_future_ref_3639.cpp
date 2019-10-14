//  Copyright (c) 2019 Arkantos493
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #3639: util::unwrapping does not work well with member functions
// see also: https://stackoverflow.com/questions/54390555/compilation-error-with-hpxdataflow-and-member-function

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>

#include <memory>
#include <vector>

class A
{
public:
    A& do_work(double x, const A& y, const A& z)
    {
        i = x + y.i + z.i;
        return *this;
    }

    double i = 1.0;
};

int main()
{
    // Create instances
    std::vector<std::unique_ptr<A>> vec;
    for (int i = 0; i < 3; ++i)
    {
        vec.emplace_back(new A());
    }

    std::vector<hpx::shared_future<A&>> a1(3);
    std::vector<hpx::shared_future<A&>> a2(3);

    // works
    a1[1] = hpx::async(&A::do_work, vec[1].get(), 1.0, *vec[0], *vec[2]);

    // compiler error here
    a2[1] = hpx::dataflow(hpx::util::unwrapping_all(&A::do_work), vec[1].get(),
        2.0, a1[0], a2[0]);

    return 0;
}
