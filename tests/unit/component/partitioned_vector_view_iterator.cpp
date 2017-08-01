//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_view.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_local_view.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/lcos/spmd_block.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);

void bulk_test( hpx::lcos::spmd_block block,
                std::size_t size_x,
                std::size_t size_y,
                std::size_t size_z,
                std::size_t elt_size,
                std::string vec_name)
{
    using const_iterator
        = typename std::vector<double>::const_iterator;
    using vector_type
        = hpx::partitioned_vector<double>;
    using view_type
        = hpx::partitioned_vector_view<double,3>;
    using view_type_iterator
        = typename view_type::iterator;
    using const_view_type_iterator
        = typename view_type::const_iterator;

    vector_type my_vector;
    my_vector.connect_to(hpx::launch::sync, vec_name);

    view_type my_view(block,
        my_vector.begin(), my_vector.end(), {size_x,size_y,size_z});

    int idx = 0;

    // Ensure that only one image is putting data into the different
    // partitions
    if(block.this_image() == 0)
    {
        // Traverse all the co-indexed elements
        for(auto i = my_view.begin(); i != my_view.end(); i++)
        {
            // It's a Put operation
            *i = std::vector<double>(elt_size,idx++);
        }

        auto left_it  = my_view.begin();
        auto right_it = my_view.cbegin();

        // Note: Useless computation, since we assign segments to themselves
        for(; left_it != my_view.end(); left_it++, right_it++)
        {
            // Check that dereferencing iterator and const_iterator does not
            // retrieve the same type
            HPX_TEST((
                !std::is_same<decltype(*left_it),decltype(*right_it)>::value));

            // It's a Put operation
            *left_it = *right_it;
        }
    }

    block.sync_all();

    if(block.this_image() == 0)
    {
        int idx = 0;

        for (std::size_t k = 0; k<size_z; k++)
            for (std::size_t j = 0; j<size_y; j++)
                for (std::size_t i = 0; i<size_x; i++)
                {
                    std::vector<double> result(elt_size,idx);

                    // It's a Get operation
                    std::vector<double> value =
                        (std::vector<double>)my_view(i,j,k);

                    const_iterator it1 = result.begin(), it2 = value.begin();
                    const_iterator end1 = result.end();

                    for (; it1 != end1; ++it1, ++it2)
                    {
                        HPX_TEST_EQ(*it1, *it2);
                    }

                    idx++;
                }

        idx = 0;

        // Re-check by traversing all the co-indexed elements
        for(auto i = my_view.cbegin(); i != my_view.cend(); i++)
        {
            std::vector<double> result(elt_size,idx);

            // It's a Get operation
            std::vector<double> value = (std::vector<double>)(*i);

            const_iterator it1 = result.begin(), it2 = value.begin();
            const_iterator end1 = result.end();

            for (; it1 != end1; ++it1, ++it2)
            {
                HPX_TEST_EQ(*it1, *it2);
            }

            idx++;
        }
    }
}
HPX_PLAIN_ACTION(bulk_test, bulk_test_action);

int main()
{
    using vector_type
        = hpx::partitioned_vector<double>;

    const std::size_t size_x = 32;
    const std::size_t size_y = 4;
    const std::size_t size_z = hpx::get_num_localities(hpx::launch::sync);

    const std::size_t elt_size = 4;
    const std::size_t num_partitions  = size_x*size_y*size_z;

    std::size_t raw_size = num_partitions*elt_size;

    vector_type my_vector(raw_size,
        hpx::container_layout( num_partitions, hpx::find_all_localities() ));

    std::string vec_name("my_vector");
    my_vector.register_as(hpx::launch::sync, vec_name);

    hpx::future<void> join =
        hpx::lcos::define_spmd_block("block", 4, bulk_test_action(),
            size_x, size_y, size_z, elt_size, vec_name);

    join.get();

    return 0;
}
