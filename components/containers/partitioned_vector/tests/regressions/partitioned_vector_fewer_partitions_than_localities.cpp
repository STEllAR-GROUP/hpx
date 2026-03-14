//  Copyright (c) 2026 Mo'men Samir
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Regression test to make sure default distribution policy works correctly
// when the number of localities is more than the number partitions

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename InIter>
void verify_values(InIter first, InIter last, T const& val)
{
    for (InIter it = first; it != last; ++it)
    {
        HPX_TEST_EQ(*it, val);
    }
}

template <typename T>
void test_with_policy(std::size_t length, std::size_t num_partitions,
    std::vector<hpx::id_type> const& localities)
{
    hpx::partitioned_vector<T> v(
        length, T(42), hpx::container_layout(num_partitions, localities));

    verify_values(v.begin(), v.end(), T(42));
    HPX_TEST_EQ(v.size(), length);
}

template <typename T>
void test_partition_locality_combinations()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t const nlocs = localities.size();
    std::size_t const length = 12;

    // fewer partitions than localities
    if (nlocs >= 2)
        test_with_policy<T>(length, nlocs - 1, localities);

    // same number of partitions as localities
    test_with_policy<T>(length, nlocs, localities);
}

int main()
{
    test_partition_locality_combinations<int>();
    test_partition_locality_combinations<double>();

    return hpx::util::report_errors();
}
#endif
