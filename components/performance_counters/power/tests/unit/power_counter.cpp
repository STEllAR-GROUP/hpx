//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_POWER_COUNTER) && !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

// clang-format off
char const* const power_counter_names[] =
{
    "/runtime/average_power",
    nullptr
};
// clang-format on

///////////////////////////////////////////////////////////////////////////////
void test_all_locality_power_counters(
    char const* const* counter_names, std::size_t locality_id)
{
    for (char const* const* p = counter_names; *p != nullptr; ++p)
    {
        // split counter type into counter path elements
        hpx::performance_counters::counter_path_elements path;
        HPX_TEST_EQ(
            hpx::performance_counters::get_counter_path_elements(*p, path),
            hpx::performance_counters::counter_status::valid_data);

        // augment the counter path elements
        path.parentinstancename_ = "locality";
        path.parentinstanceindex_ = locality_id;
        path.instancename_ = "total";
        path.instanceindex_ = -1;

        std::string name;
        HPX_TEST_EQ(hpx::performance_counters::get_counter_name(path, name),
            hpx::performance_counters::counter_status::valid_data);

        std::cout << name << '\n';

        try
        {
            hpx::performance_counters::performance_counter counter(name);
            HPX_TEST_EQ(counter.get_name(hpx::launch::sync), name);
            counter.get_value<std::size_t>(hpx::launch::sync);
        }
        catch (...)
        {
            HPX_TEST(false);    // should never happen
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_all_locality_power_counters(std::size_t locality_id)
{
    test_all_locality_thread_counters(power_counter_names, locality_id);
}

///////////////////////////////////////////////////////////////////////////////
void test_all_counters_locality(std::size_t locality_id)
{
    // locality/total
    test_all_locality_power_counters(locality_id);
}

int main()
{
    for (auto const& id : hpx::find_all_localities())
    {
        test_all_counters_locality(hpx::naming::get_locality_id_from_id(id));
    }

    return hpx::util::report_errors();
}

#else

int main()
{
    return 0;
}

#endif
