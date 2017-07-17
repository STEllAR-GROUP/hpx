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

template <typename T>
void initialize(hpx::partitioned_vector<T> & xvalues)
{
    T init_array[45] = {1,2,3,4, 5,1,2,3, 3,5,5,3, 4,2,3,2, 1,2,3,4, 5,6,5,6,
        1,2,3,4, 1,1,2,3, 4,5,4,3, 2,1,1,2,3,4, 1,2,3};
    for(int i=0; i<45; i++)
    {
        xvalues.set_value(i,init_array[i]);
    }
}

template <typename ExPolicy, typename T>
void test_find_first_of(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, std::vector<T> & sequence, int position)
{
    auto last = hpx::parallel::find_first_of(policy, xvalues.begin(),
        xvalues.end(), sequence.begin(), sequence.end());
    HPX_TEST_EQ(std::distance(xvalues.begin(), last),position);
    // printf("%d\n", (int) *last);
}

template <typename ExPolicy, typename T>
void test_find_first_of_async(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, std::vector<T> & sequence, int position)
{
    auto last = hpx::parallel::find_first_of(policy, xvalues.begin(),
        xvalues.end(), sequence.begin(), sequence.end()).get();
    HPX_TEST_EQ(std::distance(xvalues.begin(), last),position);
    // printf("Async %d\n", (int) *last);
}

template <typename T>
void find_first_of_tests(std::vector<hpx::id_type> &localities)
{
    std::size_t const num = 45;
    hpx::partitioned_vector<T> xvalues(num, hpx::container_layout(localities));
    initialize(xvalues);

    std::vector<T> sequence = {(T)1,(T)2,(T)3,(T)4};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 0);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 0);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 0);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 0);

    sequence = {(T)4,(T)5,(T)1,(T)2};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 3);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 3);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 3);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 3);

    sequence = {(T)2,(T)3,(T)3,(T)5};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 6);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 6);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 6);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 6);

    sequence = {(T)2,(T)3,(T)2,(T)1};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 13);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 13);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 13);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 13);

    sequence = {(T)3,(T)2,(T)1,(T)1};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 35);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 35);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 35);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 35);

    sequence = {(T)5,(T)6};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 20);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 20);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 20);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 20);

    sequence = {(T)3,(T)4,(T)2,(T)3,(T)2,(T)1};
    test_find_first_of(hpx::parallel::execution::seq, xvalues, sequence, 11);
    test_find_first_of(hpx::parallel::execution::par, xvalues, sequence, 11);
    test_find_first_of_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, sequence, 11);
    // test_find_first_of_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        // xvalues, sequence, 11);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    find_first_of_tests<int>(localities);
    // find_first_of_tests<double>(localities);
    return 0;
}
