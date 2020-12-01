//  Copyright (c) 2015 Steve R. Brandt
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test demonstrates the issue as reported by
//      #1787: discover_counter_types not working

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/util.hpp>

#include <iostream>

bool discover_callback(
    hpx::performance_counters::counter_info const& c, hpx::error_code&)
{
    std::cout << "counter: " << c.fullname_ << std::endl;
    return true;
}

int hpx_main()
{
    std::cout << "Counters:" << std::endl;

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    hpx::performance_counters::discover_counter_types(
        hpx::util::bind(discover_callback, _1, _2));

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    return hpx::init(argc, argv);
}
#endif
