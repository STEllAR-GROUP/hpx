//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_reduce.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <vector>
///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
T test_reduce(ExPolicy && policy,
    hpx::partitioned_vector<T> const& xvalues)
{
    return
        hpx::parallel::reduce(policy,
            xvalues.begin(), xvalues.end(),
            T(0), std::plus<T>()
        );
}

template <typename ExPolicy, typename T>
hpx::future<T>
test_reduce_async(ExPolicy && policy,
    hpx::partitioned_vector<T> const& xvalues)
{
    return
        hpx::parallel::reduce(policy,
            xvalues.begin(), xvalues.end(),
            T(0), std::plus<T>()
        );
}

template <typename T>
void reduce_tests(std::size_t num,
    hpx::partitioned_vector<T> const& xvalues)
{
    HPX_TEST_EQ(
        test_reduce(hpx::parallel::execution::seq, xvalues),
        T(num));
    HPX_TEST_EQ(
        test_reduce(hpx::parallel::execution::par, xvalues),
        T(num));

    HPX_TEST_EQ(
        test_reduce_async(
            hpx::parallel::execution::seq(hpx::parallel::execution::task),
            xvalues).get(),
        T(num));
    HPX_TEST_EQ(
        test_reduce_async(
            hpx::parallel::execution::par(hpx::parallel::execution::task),
            xvalues).get(),
        T(num));
}

template <typename T>
void reduce_tests(std::vector<hpx::id_type> &localities)
{
    std::size_t const num = 10007;
    hpx::partitioned_vector<T> xvalues(num, T(1),hpx::container_layout(localities));
    reduce_tests(num, xvalues);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    reduce_tests<int>(localities);
    reduce_tests<double>(localities);
    return 0;
}
