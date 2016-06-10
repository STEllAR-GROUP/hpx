//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void fill_vector(hpx::partitioned_vector<T>& v, T const& val)
{
    typename hpx::partitioned_vector<T>::iterator it = v.begin(), end = v.end();
    for (/**/; it != end; ++it)
        *it = val;
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void allocation_tests()
{
    std::size_t const length = 12;

    for (hpx::id_type const& locality : hpx::find_all_localities())
    {
        std::vector<hpx::compute::host::target> targets =
            hpx::compute::host::get_targets(locality).get();

        {
            hpx::partitioned_vector<T> v(length, hpx::compute::host::target_layout);
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

