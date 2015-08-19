//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that issue #1001 is resolved
// (Zero copy serialization raises assert).

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <vector>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        std::vector<int> int_vector;
        int_vector.push_back(1);
        int_vector.push_back(2);
        int_vector.push_back(3);
        std::vector<char> out_buffer;

        hpx::serialization::output_archive oa(out_buffer);
        oa << int_vector;

        hpx::serialization::input_archive ia(out_buffer, out_buffer.size());
        std::vector<int> copy_vector;
        ia >> copy_vector;

        HPX_TEST_EQ(std::size_t(3), copy_vector.size());
        HPX_TEST_EQ(copy_vector[0], 1);
        HPX_TEST_EQ(copy_vector[1], 2);
        HPX_TEST_EQ(copy_vector[2], 3);
     }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ(hpx::init(argc, argv), 0);

    return hpx::util::report_errors();
}

