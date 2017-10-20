//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

int main()
{
    using hpx::performance_counters::performance_counter;

    performance_counter average("/statistics{/runtime/uptime}/average@200");
    average.start(hpx::launch::sync);

    hpx::this_thread::sleep_for(std::chrono::seconds(1));
    double val1 = average.get_value<double>(hpx::launch::sync);
    hpx::this_thread::sleep_for(std::chrono::seconds(1));
    double val2 = average.get_value<double>(hpx::launch::sync);

    HPX_TEST_NEQ(val1, val2);

    average.stop(hpx::launch::sync);

    return hpx::util::report_errors();
}
