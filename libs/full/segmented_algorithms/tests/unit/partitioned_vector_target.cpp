//  Copyright (c) 2016-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/compute.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
using target_allocator_int = hpx::compute::host::block_allocator<int>;
using target_vector_int = hpx::compute::vector<int, target_allocator_int>;
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(int, target_vector_int)
HPX_REGISTER_PARTITIONED_VECTOR(int, target_vector_int)

using target_allocator_double = hpx::compute::host::block_allocator<double>;
using target_vector_double =
    hpx::compute::vector<double, target_allocator_double>;
HPX_REGISTER_PARTITIONED_VECTOR_DECLARATION(double, target_vector_double)
HPX_REGISTER_PARTITIONED_VECTOR(double, target_vector_double)

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void allocation_tests()
{
    using target_allocator = hpx::compute::host::block_allocator<T>;
    using target_vector = hpx::compute::vector<T, target_allocator>;

    for (hpx::id_type const& locality : hpx::find_all_localities())
    {
        std::vector<hpx::compute::host::distributed::target> targets =
            hpx::compute::host::distributed::get_targets(locality).get();

        constexpr std::size_t length = 12;
        hpx::partitioned_vector<T, target_vector> v(length, T(42),
            hpx::compute::host::target_layout(std::move(targets)));
    }

    //{
    //    hpx::partitioned_vector<T> v;
    //    copy_tests(v);
    //}
    //
    //{
    //    hpx::partitioned_vector<T> v(length);
    //    copy_tests(v);
    //}
    //
    //{
    //    hpx::partitioned_vector<T> v(length, T(42));
    //    copy_tests(v);
    //}
    //
    //copy_tests_with_policy<T>(length, 1, hpx::container_layout);
    //copy_tests_with_policy<T>(length, 3, hpx::container_layout(3));
    //copy_tests_with_policy<T>(length, 3, hpx::container_layout(3, localities));
    //copy_tests_with_policy<T>(length, localities.size(),
    //    hpx::container_layout(localities));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    allocation_tests<double>();
    allocation_tests<int>();

    return 0;
}

#endif
