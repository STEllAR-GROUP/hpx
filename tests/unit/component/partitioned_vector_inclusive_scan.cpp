//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void iota_vector(hpx::partitioned_vector<T>& v, T val)
{
    typename hpx::partitioned_vector<T>::iterator it = v.begin(), end = v.end();
    for(/**/; it != end; ++it)
        *it = val++;
}

template<typename T>
struct opt
{
    T operator()(T v1, T v2) const{
        return v1 + v2;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename DistPolicy, typename ExPolicy>
void inclusive_scan_algo_tests_with_policy(std::size_t size,
    DistPolicy const& dist_policy, ExPolicy const& policy)
{
    hpx::partitioned_vector<T> c(size, dist_policy);
    iota_vector(c, T(1));

    std::vector<T> d(c.size());
    T val(0);
    auto op =
        [val](T v1, T v2) {
            return v1 + v2;
        };

    hpx::parallel::inclusive_scan(policy,
        c.begin(), c.end(), d.begin(), val, opt<T>());

    // verify values
    std::vector<T> e(c.size());
    std::iota(e.begin(), e.end(), T(1));
    std::vector<T> f(c.size());

    hpx::parallel::v1::detail::sequential_inclusive_scan(
        e.begin(), e.end(), f.begin(), val, opt<T>());

    HPX_TEST(std::equal(d.begin(), d.end(), f.begin()));
}

template <typename T, typename DistPolicy, typename ExPolicy>
void inclusive_scan_algo_tests_with_policy_async(std::size_t size,
    DistPolicy const& dist_policy, ExPolicy const& policy)
{
    hpx::partitioned_vector<T> c(size, dist_policy);
    iota_vector(c, T(1));

    std::vector<T> d(c.size());
    T val(0);
    auto op =
        [val](T v1, T v2) {
            return v1 + v2;
        };

    auto res =
        hpx::parallel::inclusive_scan(policy,
        c.begin(), c.end(), d.begin(), val, opt<T>());
    res.get();

    // verify values
    std::vector<T> e(c.size());
    std::iota(e.begin(), e.end(), T(1));
    std::vector<T> f(c.size());

    hpx::parallel::v1::detail::sequential_inclusive_scan(
        e.begin(), e.end(), f.begin(), val, opt<T>());
    HPX_TEST(std::equal(d.begin(), d.end(), f.begin()));
}

template <typename T, typename DistPolicy>
void inclusive_scan_tests_with_policy(std::size_t size, std::size_t localities,
    DistPolicy const& policy)
{
    using namespace hpx::parallel;
    using hpx::parallel::task;

    inclusive_scan_algo_tests_with_policy<T>(size, policy, seq);
    inclusive_scan_algo_tests_with_policy<T>(size, policy, par);

    //async
    inclusive_scan_algo_tests_with_policy_async<T>(size, policy, seq(task));
    inclusive_scan_algo_tests_with_policy_async<T>(size, policy, par(task));
}


template <typename T>
void inclusive_scan_tests()
{
    std::size_t const length = 100;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    inclusive_scan_tests_with_policy<T>(length, 1, hpx::container_layout);
    inclusive_scan_tests_with_policy<T>(length, 3, hpx::container_layout(3));
    inclusive_scan_tests_with_policy<T>(length, 3, hpx::container_layout(3, localities));
    inclusive_scan_tests_with_policy<T>(length, localities.size(),
        hpx::container_layout(localities));
}
///////////////////////////////////////////////////////////////////////////////
int main()
{
    inclusive_scan_tests<int>();
//    inclusive_scan_tests<double>();

    return 0;
}

