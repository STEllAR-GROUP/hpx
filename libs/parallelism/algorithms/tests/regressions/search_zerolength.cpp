//  Copyright (c) 2015 Daniel Bourgeois
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_search.hpp>
#include <hpx/modules/testing.hpp>

#include <numeric>
#include <string>
#include <vector>

void search_zero_dist_test()
{
    using hpx::execution::par;
    using hpx::execution::seq;
    using hpx::execution::task;

    typedef std::vector<int>::iterator iterator;

    std::vector<int> c(10007);
    std::iota(c.begin(), c.end(), 1);
    std::vector<int> h(0);

    hpx::future<iterator> fut_seq =
        hpx::search(seq(task), c.begin(), c.end(), h.begin(), h.end());
    hpx::future<iterator> fut_par =
        hpx::search(par(task), c.begin(), c.end(), h.begin(), h.end());

    HPX_TEST(fut_seq.get() == c.begin());
    HPX_TEST(fut_par.get() == c.begin());
}

int hpx_main()
{
    search_zero_dist_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}
