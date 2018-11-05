//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_adjacent_difference.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct cmp
{
    cmp(T const& val = T())
      : value_(val)
    {
    }

    template <typename T_>
    bool operator()(T_ const& val) const
    {
        return val == value_;
    }

    T value_;

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        ar& value_;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
void verify_values(
    ExPolicy&& policy, hpx::partitioned_vector<T> const& v, T const& val)
{
    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;

    std::size_t size = 0;

    const_iterator end = v.end();
    for (const_iterator it = v.begin(); it != end; ++it, ++size)
    {
        HPX_TEST_EQ(*it, val);
    }

    HPX_TEST_EQ(size, v.size());
}

template <typename ExPolicy, typename T>
void verify_values_count(
    ExPolicy&& policy, hpx::partitioned_vector<T> const& v, T const& val)
{
    HPX_TEST_EQ(
        std::size_t(hpx::parallel::count(policy, v.begin(), v.end(), val)),
        v.size());
    HPX_TEST_EQ(std::size_t(hpx::parallel::count_if(
                    policy, v.begin(), v.end(), cmp<T>(val))),
        v.size());
}

template <typename ExPolicy, typename T>
void test_adjacent_difference(ExPolicy&& policy, hpx::partitioned_vector<T>& v,
    hpx::partitioned_vector<T>& w, T val)
{
    verify_values(policy, v, val);
    verify_values_count(policy, v, val);

    hpx::parallel::adjacent_difference(policy, v.begin(), v.end(), w.begin());

    verify_values(policy, w, val);
    verify_values_count(policy, w, val);
}

template <typename ExPolicy, typename T>
void verify_values_count_async(
    ExPolicy&& policy, hpx::partitioned_vector<T> const& v, T const& val)
{
    HPX_TEST_EQ(
        std::size_t(
            hpx::parallel::count(policy, v.begin(), v.end(), val).get()),
        v.size());
    HPX_TEST_EQ(std::size_t(hpx::parallel::count_if(
                    policy, v.begin(), v.end(), cmp<T>(val))
                                .get()),
        v.size());
}

template <typename ExPolicy, typename T>
void test_adjacent_difference_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& v, hpx::partitioned_vector<T>& w, T val)
{
    verify_values(policy, v, val);
    verify_values_count_async(policy, v, val);

    hpx::parallel::adjacent_difference(policy, v.begin(), v.end(), w.begin())
        .get();

    verify_values(policy, w, val);
    verify_values_count_async(policy, w, val);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void adjacent_difference_tests(std::vector<hpx::id_type>& localities)
{
    std::size_t const length = 12;

    {
        hpx::partitioned_vector<T> v;
        hpx::partitioned_vector<T> w;
        hpx::parallel::adjacent_difference(
            hpx::parallel::execution::seq, v.begin(), v.end(), w.begin());
        hpx::parallel::adjacent_difference(
            hpx::parallel::execution::par, v.begin(), v.end(), w.begin());
        hpx::parallel::adjacent_difference(
            hpx::parallel::execution::seq(hpx::parallel::execution::task),
            v.begin(), v.end(), w.begin())
            .get();
        hpx::parallel::adjacent_difference(
            hpx::parallel::execution::par(hpx::parallel::execution::task),
            v.begin(), v.end(), w.begin())
            .get();
    }

    {
        hpx::partitioned_vector<T> v(
            length, T(1), hpx::container_layout(localities));
        hpx::partitioned_vector<T> w(length, hpx::container_layout(localities));
        test_adjacent_difference(hpx::parallel::execution::seq, v, w, T(0));
        test_adjacent_difference(hpx::parallel::execution::par, v, w, T(0));
        test_adjacent_difference_async(
            hpx::parallel::execution::seq(hpx::parallel::execution::task), v, w,
            T(0));
        test_adjacent_difference_async(
            hpx::parallel::execution::par(hpx::parallel::execution::task), v, w,
            T(0));
    }

    {
        hpx::partitioned_vector<T> v(
            length, T(1), hpx::container_layout(localities));
        hpx::parallel::inclusive_scan(
            hpx::parallel::execution::seq, v.begin(), v.end(), v.begin());
        hpx::partitioned_vector<T> w(length, hpx::container_layout(localities));
        test_adjacent_difference(hpx::parallel::execution::seq, v, w, T(1));
        test_adjacent_difference(hpx::parallel::execution::par, v, w, T(1));
        test_adjacent_difference_async(
            hpx::parallel::execution::seq(hpx::parallel::execution::task), v, w,
            T(1));
        test_adjacent_difference_async(
            hpx::parallel::execution::par(hpx::parallel::execution::task), v, w,
            T(1));
    }
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    adjacent_difference_tests<double>(localities);
    return hpx::util::report_errors();
}
