//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/program_options.hpp>

#include <memory>

#include "test_common.hpp"

void stack_test_bounded()
{
    using tester_type = queue_stress_tester<true>;

    std::unique_ptr<tester_type> tester(new tester_type(2, 2));

    hpx::lockfree::stack<long> q(128);
    tester->run(q);
}

void stack_test_unbounded()
{
    using tester_type = queue_stress_tester<false>;

    std::unique_ptr<tester_type> tester(new tester_type(2, 2));

    hpx::lockfree::stack<long> q(128);
    tester->run(q);
}

void stack_test_fixed_size()
{
    using tester_type = queue_stress_tester<>;

    std::unique_ptr<tester_type> tester(new tester_type(2, 2));

    hpx::lockfree::stack<long, std::allocator<long>, 8> q;
    tester->run(q);
}

int hpx_main(hpx::program_options::variables_map&)
{
    stack_test_bounded();
    stack_test_unbounded();
    stack_test_fixed_size();

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
