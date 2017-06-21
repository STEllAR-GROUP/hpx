//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
typedef hpx::compute::host::block_allocator<int> target_allocator_int;
typedef hpx::compute::vector<int, target_allocator_int> target_vector_int;
HPX_REGISTER_PARTITIONED_VECTOR(int, target_vector_int);

typedef hpx::compute::host::block_allocator<double> target_allocator_double;
typedef hpx::compute::vector<double, target_allocator_double> target_vector_double;
HPX_REGISTER_PARTITIONED_VECTOR(double, target_vector_double);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void allocation_tests()
{
    std::size_t const length = 12;

    typedef hpx::compute::host::block_allocator<T> target_allocator;
    typedef hpx::compute::vector<T, target_allocator> target_vector;

    for (hpx::id_type const& locality : hpx::find_all_localities())
    {
        std::vector<hpx::compute::host::target> targets =
            hpx::compute::host::get_targets(locality).get();

        {
            hpx::partitioned_vector<T, target_vector> v(length, T(42),
                hpx::compute::host::target_layout);
        }
    }

//     {
//         hpx::partitioned_vector<T> v;
//         copy_tests(v);
//     }
//
//     {
//         hpx::partitioned_vector<T> v(length);
//         copy_tests(v);
//     }
//
//     {
//         hpx::partitioned_vector<T> v(length, T(42));
//         copy_tests(v);
//     }
//
//     copy_tests_with_policy<T>(length, 1, hpx::container_layout);
//     copy_tests_with_policy<T>(length, 3, hpx::container_layout(3));
//     copy_tests_with_policy<T>(length, 3, hpx::container_layout(3, localities));
//     copy_tests_with_policy<T>(length, localities.size(),
//         hpx::container_layout(localities));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    allocation_tests<double>();
    allocation_tests<int>();

    return 0;
}

