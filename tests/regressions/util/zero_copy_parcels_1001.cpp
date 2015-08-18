//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test verifies that issue #1001 is resolved
// (Zero copy serialization raises assert).

#include <hpx/hpx_init.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
hpx::serialization::serialize_buffer<int> test(hpx::serialization
    ::serialize_buffer<int> const& b)
{
    return b;
}
HPX_PLAIN_ACTION(test, test_action)

struct inc
{
    inc() : cnt_(0) {}

    int operator()()
    {
        return cnt_++;
    }

    int cnt_;
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::vector<hpx::naming::id_type> localities = hpx::find_remote_localities();

    if (localities.empty())
    {
        HPX_TEST_MSG(!localities.empty(),
            "This test must be run on more than one locality");
    }
    else
    {
        test_action act;

        std::size_t size = 1;
        for (std::size_t i = 0; i != 20; ++i)
        {
            std::vector<int> data;
            data.resize(size << i);

            std::generate(data.begin(), data.end(), inc());

            hpx::serialization::serialize_buffer<int> buffer(data.data(), data.size(),
                hpx::serialization::serialize_buffer<int>::reference);

            hpx::serialization::serialize_buffer<int> result =
                act(localities[0], buffer);

            HPX_TEST(std::equal(data.begin(), data.end(), result.data()));
        }
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

