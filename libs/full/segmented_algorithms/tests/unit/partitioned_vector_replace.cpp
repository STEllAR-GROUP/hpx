//  Copyright (c) 2026 Mo'men Samir
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_replace.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

template <typename T>
void initialize(hpx::partitioned_vector<T>& v)
{
    std::size_t index = 0;

    typename hpx::partitioned_vector<T>::iterator end = v.end();
    for (typename hpx::partitioned_vector<T>::iterator it = v.begin();
        it != end; ++it, ++index)
    {
        *it = T(index % 5);
    }
}

template <typename T>
void verify_replace(hpx::partitioned_vector<T> const& v, std::size_t first,
    std::size_t last, T const& old_value, T const& new_value)
{
    std::size_t index = 0;

    typename hpx::partitioned_vector<T>::const_iterator end = v.end();
    for (typename hpx::partitioned_vector<T>::const_iterator it = v.begin();
        it != end; ++it, ++index)
    {
        T expected = T(index % 5);
        if (index >= first && index < last && expected == old_value)
        {
            expected = new_value;
        }
        HPX_TEST_EQ(*it, expected);
    }
}

template <typename T, typename ExPolicy>
void replace_empty_range_test(ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(10, hpx::container_layout);
    initialize(c);

    hpx::replace(replace_policy, c.begin(), c.begin(), T(2), T(9));
    verify_replace(c, 0, 0, T(2), T(9));
}

template <typename T, typename ExPolicy>
void replace_single_element_test(ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(1, hpx::container_layout);

    c[0] = T(2);
    hpx::replace(replace_policy, c.begin(), c.end(), T(2), T(9));
    HPX_TEST_EQ(c[0], T(9));

    c[0] = T(4);
    hpx::replace(replace_policy, c.begin(), c.end(), T(2), T(9));
    HPX_TEST_EQ(c[0], T(4));
}

template <typename T, typename DistPolicy, typename ExPolicy>
void replace_tests_with_policy(
    std::size_t size, DistPolicy const& policy, ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(size, policy);

    initialize(c);
    hpx::replace(replace_policy, c.begin(), c.end(), T(2), T(9));
    verify_replace(c, 0, size, T(2), T(9));

    initialize(c);
    hpx::replace(replace_policy, c.begin() + 1, c.end() - 1, T(3), T(7));
    verify_replace(c, 1, size - 1, T(3), T(7));
}

template <typename T, typename DistPolicy, typename ExPolicy>
void replace_tests_with_policy_async(
    std::size_t size, DistPolicy const& policy, ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(size, policy);

    initialize(c);
    hpx::future<void> f =
        hpx::replace(replace_policy, c.begin(), c.end(), T(2), T(9));
    f.wait();
    verify_replace(c, 0, size, T(2), T(9));

    initialize(c);
    hpx::future<void> f1 =
        hpx::replace(replace_policy, c.begin() + 1, c.end() - 1, T(3), T(7));
    f1.wait();
    verify_replace(c, 1, size - 1, T(3), T(7));
}

template <typename T, typename DistPolicy>
void test_replace_with_policy(
    std::size_t size, std::size_t /* localities */, DistPolicy const& policy)
{
    using namespace hpx::execution;

    replace_tests_with_policy<T>(size, policy, seq);
    replace_tests_with_policy<T>(size, policy, par);

    replace_tests_with_policy_async<T>(size, policy, seq(task));
    replace_tests_with_policy_async<T>(size, policy, par(task));
}

template <typename T>
void replace_tests()
{
    std::size_t const length = 30;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    test_replace_with_policy<T>(length, 1, hpx::container_layout);
    test_replace_with_policy<T>(length, 3, hpx::container_layout(3));
    test_replace_with_policy<T>(
        length, 3, hpx::container_layout(3, localities));
    test_replace_with_policy<T>(
        length, localities.size(), hpx::container_layout(localities));

    using namespace hpx::execution;
    replace_empty_range_test<T>(seq);
    replace_empty_range_test<T>(par);

    replace_single_element_test<T>(seq);
    replace_single_element_test<T>(par);
}

int main()
{
    replace_tests<double>();
    replace_tests<int>();

    return hpx::util::report_errors();
}
#endif
