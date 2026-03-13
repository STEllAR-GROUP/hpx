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
struct less_than_three
{
    bool operator()(T const& value) const
    {
        return value < T(3);
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned int /* version */)
    {
        (void) ar;
    }
};

template <typename T>
void verify_replace_if(hpx::partitioned_vector<T> const& v, std::size_t first,
    std::size_t last, T const& new_value)
{
    std::size_t index = 0;

    typename hpx::partitioned_vector<T>::const_iterator end = v.end();
    for (typename hpx::partitioned_vector<T>::const_iterator it = v.begin();
        it != end; ++it, ++index)
    {
        T expected = T(index % 5);
        if (index >= first && index < last && expected < T(3))
        {
            expected = new_value;
        }
        HPX_TEST_EQ(*it, expected);
    }
}

template <typename T, typename ExPolicy>
void replace_if_empty_range_test(ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(10, hpx::container_layout);
    initialize(c);

    hpx::replace_if(
        replace_policy, c.begin(), c.begin(), less_than_three<T>(), T(8));
    verify_replace_if(c, 0, 0, T(8));
}

template <typename T, typename ExPolicy>
void replace_if_single_element_test(ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(1, hpx::container_layout);

    c[0] = T(1);
    hpx::replace_if(
        replace_policy, c.begin(), c.end(), less_than_three<T>(), T(8));
    HPX_TEST_EQ(c[0], T(8));

    c[0] = T(4);
    hpx::replace_if(
        replace_policy, c.begin(), c.end(), less_than_three<T>(), T(8));
    HPX_TEST_EQ(c[0], T(4));
}

template <typename T, typename DistPolicy, typename ExPolicy>
void replace_if_tests_with_policy(
    std::size_t size, DistPolicy const& policy, ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(size, policy);

    initialize(c);
    hpx::replace_if(
        replace_policy, c.begin(), c.end(), less_than_three<T>(), T(8));
    verify_replace_if(c, 0, size, T(8));

    initialize(c);
    hpx::replace_if(
        replace_policy, c.begin() + 2, c.end() - 2, less_than_three<T>(), T(6));
    verify_replace_if(c, 2, size - 2, T(6));
}

template <typename T, typename DistPolicy, typename ExPolicy>
void replace_if_tests_with_policy_async(
    std::size_t size, DistPolicy const& policy, ExPolicy const& replace_policy)
{
    hpx::partitioned_vector<T> c(size, policy);

    initialize(c);
    hpx::future<void> f2 = hpx::replace_if(
        replace_policy, c.begin(), c.end(), less_than_three<T>(), T(8));
    f2.wait();
    verify_replace_if(c, 0, size, T(8));

    initialize(c);
    hpx::future<void> f3 = hpx::replace_if(
        replace_policy, c.begin() + 2, c.end() - 2, less_than_three<T>(), T(6));
    f3.wait();
    verify_replace_if(c, 2, size - 2, T(6));
}

template <typename T, typename DistPolicy>
void test_replace_if_with_policy(
    std::size_t size, std::size_t /* localities */, DistPolicy const& policy)
{
    using namespace hpx::execution;

    replace_if_tests_with_policy<T>(size, policy, seq);
    replace_if_tests_with_policy<T>(size, policy, par);

    replace_if_tests_with_policy_async<T>(size, policy, seq(task));
    replace_if_tests_with_policy_async<T>(size, policy, par(task));
}

template <typename T>
void replace_if_tests()
{
    std::size_t const length = 30;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    test_replace_if_with_policy<T>(length, 1, hpx::container_layout);
    test_replace_if_with_policy<T>(length, 3, hpx::container_layout(3));
    test_replace_if_with_policy<T>(
        length, 3, hpx::container_layout(3, localities));
    test_replace_if_with_policy<T>(
        length, localities.size(), hpx::container_layout(localities));

    using namespace hpx::execution;
    replace_if_empty_range_test<T>(seq);
    replace_if_empty_range_test<T>(par);

    replace_if_single_element_test<T>(seq);
    replace_if_single_element_test<T>(par);
}

int main()
{
    replace_if_tests<double>();
    replace_if_tests<int>();

    return hpx::util::report_errors();
}
#endif
