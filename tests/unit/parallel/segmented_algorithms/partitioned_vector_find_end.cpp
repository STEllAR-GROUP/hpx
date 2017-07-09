//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_find.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>
#include <iostream>
///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
void test_find_end(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, std::vector<T> & sequence)
{
    auto last = hpx::parallel::find_end(policy, xvalues.begin(),
        xvalues.end(), sequence.begin(), sequence.end());
    HPX_TEST_EQ(*last,1);
    printf("%d\n", (int) *last);
}

template <typename ExPolicy, typename T>
void test_find_end_async(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, std::vector<T> & sequence)
{
    auto last = hpx::parallel::find_end(policy, xvalues.begin(),
        xvalues.end(), sequence.begin(), sequence.end()).get();
    HPX_TEST_EQ(*last,1);
    printf("Async %d\n", (int) *last);
}

template <typename T>
void initialize(hpx::partitioned_vector<T> & xvalues);

template <typename T>
void find_end_tests(std::vector<hpx::id_type> &localities)
{
    std::size_t const num = 42;
    hpx::partitioned_vector<T> xvalues(num, hpx::container_layout(localities));
    initialize(xvalues);
    std::vector<T> sequence = {1,2,3,4};
    test_find_end(hpx::parallel::execution::seq, xvalues, sequence);
    test_find_end(hpx::parallel::execution::par, xvalues, sequence);
    test_find_end_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence);
    test_find_end_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        xvalues, sequence);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    find_end_tests<int>(localities);
    // find_end_tests<double>(localities);
    return 0;
}

template <typename T>
void initialize(hpx::partitioned_vector<T> & xvalues)
{
  xvalues.set_value(0, 1);
  xvalues.set_value(1, 2);
  xvalues.set_value(2, 3);
  xvalues.set_value(3, 4);
  xvalues.set_value(4, 5);
  xvalues.set_value(5, 1);
  xvalues.set_value(6, 2);
  xvalues.set_value(7, 3);
  xvalues.set_value(8, 3);
  xvalues.set_value(9, 5);
  xvalues.set_value(10, 5);
  xvalues.set_value(11, 3);
  xvalues.set_value(12, 4);
  xvalues.set_value(13, 2);
  xvalues.set_value(14, 3);
  xvalues.set_value(15, 2);
  xvalues.set_value(16, 1);
  xvalues.set_value(17, 2);
  xvalues.set_value(18, 3);
  xvalues.set_value(19, 4);
  xvalues.set_value(20, 1);
  xvalues.set_value(21, 2);
  xvalues.set_value(22, 3);
  xvalues.set_value(23, 4);
  xvalues.set_value(24, 1);
  xvalues.set_value(25, 2);
  xvalues.set_value(26, 3);
  xvalues.set_value(27, 4);
  xvalues.set_value(28, 1);
  xvalues.set_value(29, 1);
  xvalues.set_value(30, 2);
  xvalues.set_value(31, 3);
  xvalues.set_value(32, 4);
  xvalues.set_value(33, 5);
  xvalues.set_value(34, 4);
  xvalues.set_value(35, 3);
  xvalues.set_value(36, 2);
  xvalues.set_value(37, 1);
  xvalues.set_value(38, 1);
  xvalues.set_value(39, 2);
  xvalues.set_value(40, 3);
  xvalues.set_value(41, 4);
}
